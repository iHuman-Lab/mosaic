"""Run on the LAPTOP the Tobii eye tracker is plugged into. Publishes the gaze as an LSL stream.

    pip install tobii-research pylsl
    python tobii_to_lsl.py --width 1920 --height 1080

It uses the same stream name, type and 9 channels as MOSAIC's TobiiEyeTracker, so recordings from
either source are read by the same code in the notebooks. Leave it running, start LabRecorder, then
run your task (``shasta_gui_lsl.py`` or MOSAIC). Gaze is published in pixels of the screen size you give.

Tested with a Tobii Pro Spark on Ubuntu 22.04: the stream appears at 60 Hz and Step 8 of
``02_mosaic_human_ai.ipynb`` records it with ``eye_tracker.source: live``. It does not calibrate; the
tracker uses whatever calibration it already holds.
"""

import argparse
import math
import time

import pylsl
import tobii_research as tr

from lsl_tools import GAZE_CHANNELS, GAZE_UNITS

NAN = float("nan")


def _mean(a, b):
    vals = [v for v in (a, b) if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else NAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True, help="screen width in pixels")
    parser.add_argument("--height", type=int, required=True, help="screen height in pixels")
    parser.add_argument("--name", default="TobiiEyeTracker")
    args = parser.parse_args()

    trackers = tr.find_all_eyetrackers()
    if not trackers:
        raise SystemExit("No Tobii eye tracker found")
    tracker = trackers[0]
    print("Using", tracker.model, tracker.serial_number)

    info = pylsl.StreamInfo(args.name, "Gaze", len(GAZE_CHANNELS), 60, "float32", "tobii_eye_tracker")
    channels = info.desc().append_child("channels")
    for label, unit in zip(GAZE_CHANNELS, GAZE_UNITS):
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", unit)
    outlet = pylsl.StreamOutlet(info)

    def on_gaze(g):
        left, right = g["left_gaze_point_on_display_area"], g["right_gaze_point_on_display_area"]
        x, y = _mean(left[0], right[0]), _mean(left[1], right[1])
        origin_l = g["left_gaze_origin_in_user_coordinate_system"]
        origin_r = g["right_gaze_origin_in_user_coordinate_system"]
        eye = [_mean(origin_l[i], origin_r[i]) for i in range(3)]
        valid = g["left_gaze_point_validity"] + 2 * g["right_gaze_point_validity"]
        outlet.push_sample([
            g["device_time_stamp"],
            x * args.width if not math.isnan(x) else NAN,
            y * args.height if not math.isnan(y) else NAN,
            _mean(g["left_pupil_diameter"], g["right_pupil_diameter"]),
            *eye, eye[2] / 10.0 if not math.isnan(eye[2]) else NAN, float(valid),
        ])

    tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, on_gaze, as_dictionary=True)
    print("Streaming gaze to LSL. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, on_gaze)


if __name__ == "__main__":
    main()
