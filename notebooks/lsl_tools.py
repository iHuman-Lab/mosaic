"""Shared helpers for the SHASTA and MOSAIC tutorial notebooks.

Three small pieces, used by both notebooks:

* **Streams**  : ``MarkerOutlet`` (events as text) and ``SyntheticGaze`` (a stand-in eye tracker
  that publishes the same 9-channel layout as the real Tobii stream).
* **Recording**: ``Recorder`` pulls every LSL stream you name and saves one ``.xdf`` file
  (a minimal stand-in for LabRecorder, so a hub with no LabRecorder can still make XDF files).
* **Analysis** : ``load_streams`` (read XDF with pyxdf), ``gaze_table``, ``marker_table``,
  ``study_tables`` (the same two tables from a study-runner recording), ``detect_fixations`` and a
  few plotting helpers.
"""

import json
import math
import re
import struct
import threading
import time

import numpy as np
import pylsl

# Channel layout of the Tobii gaze stream used in MOSAIC (experiment/sensors/eye_tracker/tobii.py)
GAZE_CHANNELS = [
    "device_timestamp", "avg_gaze_point_x", "avg_gaze_point_y", "avg_pupil_diam",
    "avg_eye_pos_x", "avg_eye_pos_y", "avg_eye_pos_z", "avg_eye_distance", "eye_validities",
]
GAZE_UNITS = ["us", "px", "px", "mm", "mm", "mm", "mm", "cm", "code"]


# --------------------------------------------------------------------------------------
# Streams
# --------------------------------------------------------------------------------------
class MarkerOutlet:
    """Publish events as an irregular, text-valued LSL stream (type ``Markers``)."""

    def __init__(self, name="HSI-Events", source_id=None):
        self.source_id = source_id or f"{name}-{time.time_ns()}"      # unique, so a recorder can pick this stream out
        info = pylsl.StreamInfo(name, "Markers", 1, pylsl.IRREGULAR_RATE, "string", self.source_id)
        self.outlet = pylsl.StreamOutlet(info)

    def push(self, event, **fields):
        """Send one event; extra keyword arguments are stored as JSON next to it."""
        self.outlet.push_sample([json.dumps({"event": event, **fields})])

    def close(self):
        """Take the stream off the network."""
        self.outlet = None


def attach_markers(commander, outlet):
    """Make a SHASTA ``SwarmCommander`` send every log line to ``outlet`` as a marker."""
    original = commander.log

    def log(text):
        original(text)
        outlet.push("log", text=text, step=commander.step_count)

    commander.log = log
    return commander


def screen_xy(view, xy):
    """Map position (metres) -> screen pixel ``(x, y)``. ``view`` is the GUI's ``MapView``."""
    sx, sy = np.asarray(view.to_screen(xy))[0]
    return float(sx), float(sy)


def attach_order_markers(commander, outlet, view):
    """Send an ``order`` marker for every order given, with the target's position on screen (px).

    ``view`` is the GUI's ``MapView`` (``gui.view``); its ``to_screen`` turns map metres into pixels, so the
    marker says where the human *should* be looking after the order. Gaze can then be compared with it."""
    original = commander._order

    def _order(group_id, node):
        original(group_id, node)
        sx, sy = screen_xy(view, commander.node_position(node))
        outlet.push("order", group=int(group_id), node=int(node), screen_x=sx, screen_y=sy,
                    step=commander.step_count)

    commander._order = _order
    return commander


class SyntheticGaze:
    """A stand-in eye tracker: same 9-channel ``Gaze`` stream as the real Tobii, but made up.

    The gaze point chases ``look_at(x, y)`` (pixels) with a quick jump (saccade), small tremor and
    the odd blink. The stream header says ``synthetic = true`` so a file is never mistaken for real data.
    """

    def __init__(self, width=1280, height=800, rate=60, name="TobiiEyeTracker", seed=0):
        self.width, self.height, self.rate = width, height, rate
        self.source_id = f"synthetic-gaze-{time.time_ns()}"
        info = pylsl.StreamInfo(name, "Gaze", len(GAZE_CHANNELS), rate, "float32", self.source_id)
        desc = info.desc()
        desc.append_child_value("synthetic", "true")
        channels = desc.append_child("channels")
        for label, unit in zip(GAZE_CHANNELS, GAZE_UNITS):
            channel = channels.append_child("channel")
            channel.append_child_value("label", label)
            channel.append_child_value("unit", unit)
        self.outlet = pylsl.StreamOutlet(info)
        self.target = np.array([width / 2, height / 2], dtype=float)
        self.pos = self.target.copy()
        self.rng = np.random.default_rng(seed)
        self._stop = threading.Event()
        self._thread = None

    def look_at(self, x, y):
        self.target = np.array([x, y], dtype=float)

    def _run(self):
        period = 1.0 / self.rate
        next_time = time.perf_counter()
        blink_until = 0.0
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_time:
                time.sleep(min(next_time - now, 0.002))
                continue
            next_time += period
            self.pos += 0.35 * (self.target - self.pos)            # saccade: fast catch-up
            tremor = self.rng.normal(0, 1.5, size=2)               # fixational jitter, px
            x, y = self.pos + tremor
            if now > blink_until and self.rng.random() < 0.002:
                blink_until = now + 0.15
            valid = now > blink_until
            pupil = 3.2 + 0.1 * math.sin(now) + self.rng.normal(0, 0.02)
            sample = [
                pylsl.local_clock() * 1e6,
                x if valid else float("nan"), y if valid else float("nan"),
                pupil if valid else float("nan"),
                0.0, 0.0, 600.0, 60.0, 0.0 if valid else 4.0,
            ]
            self.outlet.push_sample(sample)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop the gaze and take the stream off the network."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        self.outlet = None


# --------------------------------------------------------------------------------------
# XDF writing (the small subset of the format that pyxdf and LabRecorder files use)
# --------------------------------------------------------------------------------------
_STRUCT = {"float32": "f", "double64": "d", "int32": "i", "int16": "h", "int8": "b"}


def _varlen(n):
    if n < 256:
        return b"\x01" + struct.pack("<B", n)
    if n < 2 ** 32:
        return b"\x04" + struct.pack("<I", n)
    return b"\x08" + struct.pack("<Q", n)


def _chunk(tag, payload):
    body = struct.pack("<H", tag) + payload
    return _varlen(len(body)) + body


class XDFWriter:
    """Write streams to an ``.xdf`` file. Samples can be appended in several batches."""

    def __init__(self, path):
        self.file = open(path, "wb")
        self.file.write(b"XDF:")
        self.file.write(_chunk(1, b'<?xml version="1.0"?><info><version>1.0</version></info>'))
        self.formats = {}

    def add_stream(self, stream_id, info_xml, channel_format):
        self.formats[stream_id] = channel_format
        self.file.write(_chunk(2, struct.pack("<I", stream_id) + info_xml.encode("utf-8")))

    def add_samples(self, stream_id, timestamps, samples):
        if not len(timestamps):
            return
        fmt = self.formats[stream_id]
        out = bytearray(struct.pack("<I", stream_id) + _varlen(len(timestamps)))
        for t, sample in zip(timestamps, samples):
            out += b"\x08" + struct.pack("<d", t)
            if fmt == "string":
                for value in sample:
                    raw = str(value).encode("utf-8")
                    out += _varlen(len(raw)) + raw
            else:
                out += struct.pack("<" + _STRUCT[fmt] * len(sample), *sample)
        self.file.write(_chunk(3, bytes(out)))

    def add_clock_offset(self, stream_id, time_stamp, offset=0.0):
        self.file.write(_chunk(4, struct.pack("<Idd", stream_id, time_stamp, offset)))

    def close_stream(self, stream_id, first, last, count):
        xml = (f"<?xml version=\"1.0\"?><info><first_timestamp>{first}</first_timestamp>"
               f"<last_timestamp>{last}</last_timestamp><sample_count>{count}</sample_count>"
               f"<clock_offsets/></info>")
        self.file.write(_chunk(6, struct.pack("<I", stream_id) + xml.encode("utf-8")))

    def close(self):
        self.file.close()


# --------------------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------------------
class Recorder:
    """Record LSL streams to one XDF file. ``Recorder(path, ["Markers", "Gaze"])`` records by stream type.

    Every stream of those types on the network is recorded, including other people's on a shared machine.
    Pass ``source_ids`` to keep only the streams with those ids."""

    def __init__(self, path, stream_types=("Markers", "Gaze"), wait=3.0, source_ids=None):
        self.path = path
        self.inlets = []
        flags = pylsl.proc_clocksync | pylsl.proc_dejitter
        if source_ids is None:
            found = [info for stream_type in stream_types for info in pylsl.resolve_byprop("type", stream_type, timeout=wait)]
        else:                                      # ask for each stream by its id, so someone else's is never picked up
            found = [info for source_id in source_ids for info in pylsl.resolve_byprop("source_id", source_id, timeout=wait)]
            found = [info for info in found if info.type() in stream_types]
        for info in found:
            inlet = pylsl.StreamInlet(info, max_buflen=360, processing_flags=flags)
            inlet.open_stream(timeout=5)           # connect now, so no early samples are lost
            self.inlets.append(inlet)
        if not self.inlets:
            raise RuntimeError(f"No LSL streams found for types {list(stream_types)}")
        self.streams = [(inlet.info().name(), inlet.info().type()) for inlet in self.inlets]
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        writer = XDFWriter(self.path)
        meta = []
        for index, inlet in enumerate(self.inlets, start=1):
            info = inlet.info()
            fmt = {pylsl.cf_string: "string", pylsl.cf_float32: "float32", pylsl.cf_double64: "double64",
                   pylsl.cf_int32: "int32"}[info.channel_format()]
            writer.add_stream(index, inlet.info(timeout=5).as_xml(), fmt)
            meta.append({"first": None, "last": None, "count": 0})
        while not self._stop.is_set() or self._pending():
            for index, inlet in enumerate(self.inlets, start=1):
                samples, stamps = inlet.pull_chunk(timeout=0.0, max_samples=1024)
                if stamps:
                    writer.add_samples(index, stamps, samples)
                    m = meta[index - 1]
                    m["first"] = stamps[0] if m["first"] is None else m["first"]
                    m["last"] = stamps[-1]
                    m["count"] += len(stamps)
            time.sleep(0.01)
        for index, m in enumerate(meta, start=1):
            for stamp in (m["first"], m["last"]):
                writer.add_clock_offset(index, stamp or 0.0)
            writer.close_stream(index, m["first"] or 0, m["last"] or 0, m["count"])
        writer.close()

    def _pending(self):
        return any(inlet.samples_available() for inlet in self.inlets)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        time.sleep(0.3)           # let the last samples arrive
        self._stop.set()
        self._thread.join(timeout=10)
        return self.path


# --------------------------------------------------------------------------------------
# Reading and analysis
# --------------------------------------------------------------------------------------
def load_streams(path):
    """Read an XDF file. Returns ``{stream_type: dict(name, type, ts, data, labels, synthetic)}``."""
    import pyxdf

    streams, _ = pyxdf.load_xdf(path)
    out = {}
    for stream in streams:
        if len(stream["time_stamps"]) == 0:        # a stream that was listed but sent nothing
            continue
        info = stream["info"]
        desc = info["desc"][0] if isinstance(info.get("desc"), list) and info["desc"] and info["desc"][0] else {}
        labels = []
        try:
            labels = [c["label"][0] for c in desc["channels"][0]["channel"]]
        except (KeyError, TypeError, IndexError):
            pass
        synthetic = str(desc.get("synthetic", ["false"])[0]).lower() == "true" if isinstance(desc, dict) else False
        out[info["type"][0]] = {
            "name": info["name"][0], "type": info["type"][0], "ts": np.asarray(stream["time_stamps"]),
            "data": stream["time_series"], "labels": labels, "synthetic": synthetic,
        }
    return out


def gaze_table(stream):
    """Gaze stream -> dict of arrays: ``t`` (seconds), ``x``, ``y`` (px), ``pupil``, ``valid`` (bool)."""
    labels = stream["labels"] or GAZE_CHANNELS
    col = {name: i for i, name in enumerate(labels)}
    data = np.asarray(stream["data"], dtype=float)
    x, y = data[:, col["avg_gaze_point_x"]], data[:, col["avg_gaze_point_y"]]
    return {
        "t": stream["ts"], "x": x, "y": y, "pupil": data[:, col["avg_pupil_diam"]],
        "valid": ~(np.isnan(x) | np.isnan(y)),
    }


_ORDER = re.compile(r"Group (\d+) ordered to node (\d+)")


def marker_table(stream):
    """Marker stream -> list of dicts ``{t, event, ...}`` (JSON payload merged into each row)."""
    rows = []
    for t, sample in zip(stream["ts"], stream["data"]):
        row = {"t": float(t)}
        try:
            row.update(json.loads(sample[0]))
        except (ValueError, TypeError):
            row["event"] = str(sample[0])
        rows.append(row)
    return rows


_STATE_FIELDS = ("step_count", "agent_x", "agent_y", "saved_victims", "remaining_victims", "action", "reward")


def fullscreen_aois(screen):
    """The three screen areas ``{name: [x, y, width, height]}`` (px) of the fullscreen study GUI.

    The study runner makes the game view as tall as the screen, puts a panel half as wide beside it
    and centres the two (``SAREnvGUI._calculate_offsets``); ``screen = (width, height)`` in pixels."""
    width, height = screen
    scale = min(width / (height + height // 2), 1.0)
    game, panel = height * scale, (height // 2) * scale
    left, top = (width - game - panel) / 2, (height - game) / 2
    return {"game": [left, top, game, game],
            "info": [left + game, top, panel, game / 2],
            "chat": [left + game, top + game / 2, panel, game / 2]}


def study_tables(streams, screen):
    """Gaze and game tables from a study-runner recording (MOSAIC's ``experiment`` package and LabRecorder).

    That file differs from the one this notebook records in two ways: the game is a ``GameState`` stream
    (one JSON state per frame) instead of ``Markers``, and gaze is in fractions of the screen (0..1)
    instead of pixels. Returns ``(gaze, rows)`` in the form ``gaze_table`` and ``marker_table`` give:
    gaze in pixels of ``screen = (width, height)``, and rows with one ``session_start``, a ``state`` per
    game step and a ``rescue`` whenever ``saved_victims`` goes up."""
    gaze = gaze_table(streams["Gaze"])
    gaze["x"], gaze["y"] = gaze["x"] * screen[0], gaze["y"] * screen[1]

    game = streams["GameState"]
    rows = [{"t": float(game["ts"][0]), "event": "session_start", "screen": list(screen),
             "aois": fullscreen_aois(screen), "synthetic_gaze": False}]
    steps = saved = None
    for t, sample in zip(game["ts"], game["data"]):
        state = json.loads(sample[0])
        if state["total_steps"] != steps:
            steps = state["total_steps"]
            rows.append({"t": float(t), "event": "state", **{key: state[key] for key in _STATE_FIELDS}})
        if saved is not None and state["saved_victims"] > saved:
            rows.append({"t": float(t), "event": "rescue"})
        saved = state["saved_victims"]
    return gaze, rows


def gaze_area(gaze, aois):
    """Which area each gaze sample is in: an array of names, one per sample.

    A name from ``aois`` (``{name: [x, y, width, height]}`` in px), ``"elsewhere"`` for valid gaze outside
    them all, or ``"no data"`` where the tracker had no valid gaze. ``gaze`` is a ``gaze_table`` dict."""
    x, y, valid = gaze["x"], gaze["y"], gaze["valid"]
    area = np.where(valid, "elsewhere", "no data").astype(object)
    for name, (left, top, width, height) in aois.items():
        area[valid & (x >= left) & (x < left + width) & (y >= top) & (y < top + height)] = name
    return area


def detect_fixations(t, x, y, max_dispersion=40.0, min_duration=0.10):
    """Dispersion-threshold (I-DT) fixations: a list of ``dict(start, end, x, y)`` (px, seconds)."""
    ok = ~(np.isnan(x) | np.isnan(y))
    t, x, y = t[ok], x[ok], y[ok]
    fixations, i, n = [], 0, len(t)
    while i < n:
        j = i
        while j < n and t[j] - t[i] < min_duration:
            j += 1
        if j >= n:
            break
        window = slice(i, j + 1)
        if (x[window].max() - x[window].min()) + (y[window].max() - y[window].min()) <= max_dispersion:
            while j + 1 < n:
                w = slice(i, j + 2)
                if (x[w].max() - x[w].min()) + (y[w].max() - y[w].min()) > max_dispersion:
                    break
                j += 1
            fixations.append({"start": t[i], "end": t[j], "x": float(x[i:j + 1].mean()), "y": float(y[i:j + 1].mean())})
            i = j + 1
        else:
            i += 1
    return fixations


def gaze_heatmap(x, y, shape, sigma=25):
    """Smoothed 2-D histogram of gaze points on a screen of ``shape = (height, width)``."""
    ok = ~(np.isnan(x) | np.isnan(y))
    heat, _, _ = np.histogram2d(y[ok], x[ok], bins=(shape[0] // 8, shape[1] // 8),
                                range=[[0, shape[0]], [0, shape[1]]])
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(heat, sigma / 8)
