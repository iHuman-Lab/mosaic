"""
inspect_xdf.py  –  Explore an unknown XDF file.

Usage:
    python inspect_xdf.py <file.xdf>                        # summary of all streams
    python inspect_xdf.py <file.xdf> --stream 2             # full detail for stream #2
    python inspect_xdf.py <file.xdf> --markers              # dump all marker/event values
    python inspect_xdf.py <file.xdf> --save-config          # write stream mapping to config.yaml
    python inspect_xdf.py <file.xdf> --save-config --config path/to/config.yaml
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyxdf
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(info: dict, key: str, fallback: str = "?") -> str:
    val = info.get(key, [fallback])
    return val[0] if isinstance(val, list) else str(val)


def _is_marker_stream(info: dict) -> bool:
    t   = _get(info, "type").lower()
    n   = _get(info, "name").lower()
    fmt = _get(info, "channel_format").lower()
    return "marker" in t or "marker" in n or "string" in t or fmt == "string"


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(streams: list[dict]) -> None:
    header = f"\n{'#':>3}  {'Name':<30}  {'Type':<18}  {'Ch':>4}  {'Hz':>8}  {'Samples':>8}  {'Format':<10}"
    print(header)
    print("-" * len(header))
    for i, s in enumerate(streams):
        info = s["info"]
        name     = _get(info, "name")
        stype    = _get(info, "type")
        channels = _get(info, "channel_count")
        srate    = _get(info, "nominal_srate")
        fmt      = _get(info, "channel_format")
        n_samp   = len(s.get("time_stamps", []))
        marker   = "  <-- MARKER" if _is_marker_stream(info) else ""
        print(f"{i:>3}  {name:<30}  {stype:<18}  {channels:>4}  {srate:>8}  {n_samp:>8}  {fmt:<10}{marker}")


# ---------------------------------------------------------------------------
# Detailed view for one stream
# ---------------------------------------------------------------------------

def print_stream_detail(s: dict, idx: int) -> None:
    info = s["info"]
    print(f"\n{'='*60}")
    print(f"Stream #{idx}  —  {_get(info, 'name')}")
    print(f"{'='*60}")

    # --- metadata ---
    for key in ("type", "channel_count", "nominal_srate", "channel_format",
                "source_id", "version"):
        print(f"  {key:<20}: {_get(info, key)}")

    # --- channel labels ---
    desc = info.get("desc", [{}])
    if isinstance(desc, list) and desc:
        desc = desc[0]
    channels_node = desc.get("channels", [{}])
    if isinstance(channels_node, list) and channels_node:
        channels_node = channels_node[0]
    ch_list = channels_node.get("channel", [])
    if ch_list:
        print(f"\n  Channel labels ({len(ch_list)}):")
        for j, ch in enumerate(ch_list):
            label = ch.get("label", ["?"])[0] if isinstance(ch.get("label"), list) else ch.get("label", "?")
            unit  = ch.get("unit",  [""])[0]  if isinstance(ch.get("unit"),  list) else ch.get("unit",  "")
            print(f"    [{j:>3}]  {label}  {('(' + unit + ')') if unit else ''}")

    # --- time range ---
    ts = s.get("time_stamps", [])
    if len(ts):
        print(f"\n  Time range : {ts[0]:.3f} – {ts[-1]:.3f} s  ({ts[-1]-ts[0]:.1f} s total)")
        print(f"  Samples    : {len(ts)}")

    # --- data preview ---
    series = s.get("time_series", [])
    if len(series):
        arr = np.array(series)
        print(f"\n  Data shape : {arr.shape}")
        if arr.dtype.kind in ("f", "i", "u"):
            print(f"  Min / Max  : {arr.min():.4g} / {arr.max():.4g}")
            print(f"  Mean / Std : {arr.mean():.4g} / {arr.std():.4g}")
        print("\n  First 5 samples:")
        for row in series[:5]:
            print(f"    {row}")


# ---------------------------------------------------------------------------
# Marker dump
# ---------------------------------------------------------------------------

def print_all_markers(streams: list[dict]) -> None:
    rows = []
    for s in streams:
        info = s["info"]
        if not _is_marker_stream(info):
            continue
        stream_name = _get(info, "name")
        for t, v in zip(s.get("time_stamps", []), s.get("time_series", [])):
            value = v[0] if isinstance(v, (list, tuple, np.ndarray)) else v
            rows.append((float(t), stream_name, str(value)))

    if not rows:
        print("\n[WARN] No marker streams detected.")
        return

    rows.sort(key=lambda r: r[0])
    print(f"\n{'Timestamp':>14}  {'Stream':<30}  Value")
    print("-" * 80)
    for t, stream_name, value in rows:
        print(f"{t:>14.3f}  {stream_name:<30}  {value}")

    # unique values
    unique = sorted({r[2] for r in rows})
    print(f"\nUnique marker values ({len(unique)}):")
    for v in unique:
        print(f"  {v}")


# ---------------------------------------------------------------------------
# Config writer
# ---------------------------------------------------------------------------

def _detect_streams(streams: list[dict]) -> dict:
    """Auto-detect game and eyetracker stream names and llm_field from stream list."""
    game_stream = None
    eye_stream  = None

    for s in streams:
        info = s["info"]
        fmt  = _get(info, "channel_format").lower()
        srate = float(_get(info, "nominal_srate", "0") or "0")
        name  = _get(info, "name")

        if fmt == "string":
            game_stream = name
        elif fmt == "float32" and srate > 0:
            eye_stream = name

    # Detect llm_field by parsing first JSON sample from game stream
    llm_field = "llm_model"
    if game_stream:
        for s in streams:
            if _get(s["info"], "name") == game_stream:
                series = s.get("time_series", [])
                if series:
                    try:
                        sample = json.loads(series[0][0])
                        # pick known LLM-related keys if present
                        for candidate in ("llm_model", "llm_provider", "model"):
                            if candidate in sample:
                                llm_field = candidate
                                break
                        # also collect unique llm_model values
                        counts = Counter(
                            json.loads(v[0]).get(llm_field, "?")
                            for v in series
                        )
                        print(f"\nDetected '{llm_field}' values: {dict(counts)}")
                    except (json.JSONDecodeError, IndexError):
                        pass
                break

    return {"game": game_stream, "eyetracker": eye_stream, "llm_field": llm_field}


def save_config(detected: dict, config_path: Path) -> None:
    """Write detected stream mapping into config.yaml under the 'streams:' key."""
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    cfg["streams"] = detected

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\nSaved stream mapping to {config_path}:")
    for k, v in detected.items():
        print(f"  {k}: {v!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def inspect(xdf_path: str | Path,
            stream_idx: int | None = None,
            show_markers: bool = False,
            save_cfg: bool = False,
            config_path: Path | None = None) -> None:
    path = Path(xdf_path)
    if not path.exists():
        raise FileNotFoundError(f"XDF file not found: {path}")

    print(f"\nLoading: {path}")
    streams, _ = pyxdf.load_xdf(str(path))
    print(f"Streams found: {len(streams)}")

    print_summary(streams)

    if save_cfg:
        detected = _detect_streams(streams)
        cfg_path = config_path or Path(__file__).parent / "config.yaml"
        save_config(detected, cfg_path)
    elif show_markers:
        print_all_markers(streams)
    elif stream_idx is not None:
        if stream_idx < 0 or stream_idx >= len(streams):
            print(f"\n[ERROR] Stream index {stream_idx} out of range (0–{len(streams)-1})")
        else:
            print_stream_detail(streams[stream_idx], stream_idx)
    else:
        print("\nTips:")
        print("  --stream <N>        full metadata + stats for stream #N")
        print("  --markers           dump every marker/event value")
        print("  --save-config       write stream mapping into config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect an unknown XDF file.")
    parser.add_argument("xdf", help="Path to the XDF file")
    parser.add_argument("--stream", type=int, default=None,
                        help="Show full detail for this stream index")
    parser.add_argument("--markers", action="store_true",
                        help="Dump all marker/event values (including LLM condition names)")
    parser.add_argument("--save-config", action="store_true",
                        help="Auto-detect stream names and write them to config.yaml")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.yaml (default: statistics/config.yaml)")
    args = parser.parse_args()
    inspect(args.xdf, args.stream, args.markers, args.save_config, args.config)
