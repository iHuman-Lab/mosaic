from __future__ import annotations

import math
import queue
from typing import Any

import tobii_research as tobii
from ixp.sensors.base_sensor import Sensor
from psychopy import core, visual

from .data import (
    get_3d_position,
    get_eye_distance,
    get_eye_validity,
    get_gaze_position,
    get_pupil_size,
    get_trackbox_position,
)
from .utils import (
    active_disp_to_mont_pix,
    detect_screen_size_mm,
    print_calibration_results,
    show_calibration_point,
)


class TobiiEyeTracker(Sensor):
    """
    Interface for Tobii eye tracking devices.

    Parameters
    ----------
    config : dict, optional
        Overrides for the default LSL stream settings. Keys: name, type,
        channel_count (9), nominal_srate (60), channel_format, source_id,
        serial_string (to target a specific device).

    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        default_config = {
            "name": "TobiiEyeTracker",
            "type": "Gaze",
            "channel_count": 9,
            "nominal_srate": 60,
            "channel_format": "float32",
            "source_id": "tobii_eye_tracker",
        }
        if config:
            default_config.update(config)

        super().__init__(default_config)

        self.eyetracker = None
        self.tracking = False
        self.gaze_data: dict[str, Any] = {}
        self.win = None
        self._gaze_queue: queue.Queue = queue.Queue()

    def initialize(self) -> None:
        """Connect to the tracker and start gaze streaming."""
        serial_string = self.config.get("serial_string")
        self.connect_to_tracker(serial_string)
        self.start_tracking()

    def get_data_signature(self) -> dict[str, Any]:
        """Return channel names and units for the 9-channel gaze stream."""
        return {
            "channels": [
                "device_timestamp",
                "avg_gaze_point_x",
                "avg_gaze_point_y",
                "avg_pupil_diam",
                "avg_eye_pos_x",
                "avg_eye_pos_y",
                "avg_eye_pos_z",
                "avg_eye_distance",
                "eye_validities",
            ],
            "units": ["us", "px", "px", "mm", "mm", "mm", "mm", "cm", "code"],
        }

    def read_data(self) -> list[float] | None:
        """
        Return 9-channel gaze sample or None if no data.

        Channels: device_timestamp, avg_gaze_point_x, avg_gaze_point_y, avg_pupil_diam,
        avg_eye_pos_x, avg_eye_pos_y, avg_eye_pos_z, avg_eye_distance, eye_validities.
        """
        try:
            gaze_data = self._gaze_queue.get_nowait()
        except queue.Empty:
            return None

        timestamp = gaze_data.get("device_time_stamp", 0.0)

        gaze_pos = get_gaze_position(gaze_data)
        if self.win is not None and not any(math.isnan(v) for v in gaze_pos):
            gaze_xy = active_disp_to_mont_pix(gaze_pos, self.win)
        else:
            gaze_xy = gaze_pos

        avg_pupil = get_pupil_size(gaze_data)
        eye_pos = get_3d_position(gaze_data)
        eye_dist = get_eye_distance(eye_pos)
        validities = get_eye_validity(gaze_data)

        return [
            timestamp,
            gaze_xy[0],
            gaze_xy[1],
            avg_pupil,
            eye_pos[0],
            eye_pos[1],
            eye_pos[2],
            eye_dist,
            float(validities),
        ]

    def connect_to_tracker(self, serial_string: str | None = None) -> None:
        """Connect to the first found tracker, or to the one matching serial_string."""
        trackers = tobii.find_all_eyetrackers()
        if not trackers:
            msg = "No eye trackers found"
            raise ValueError(msg)

        if serial_string:
            selected_tracker = next(
                (t for t in trackers if t.serial_number == serial_string), trackers[0]
            )
        else:
            selected_tracker = trackers[0]

        try:
            self.eyetracker = tobii.EyeTracker(selected_tracker.address)
            self.logger.info(
                "Connected to %s (Model: %s, S/N: %s)",
                selected_tracker.device_name,
                selected_tracker.model,
                selected_tracker.serial_number,
            )
        except ConnectionError as e:
            msg = f"Failed to connect: {e}"
            raise ConnectionError(msg) from e

        self._apply_sampling_rate(self.config["nominal_srate"])

    def _apply_sampling_rate(self, desired_hz: float) -> None:
        """Set tracker output frequency to desired_hz if supported; warn and use closest otherwise."""
        supported = self.eyetracker.get_all_gaze_output_frequencies()
        if not supported:
            return

        if desired_hz in supported:
            self.eyetracker.set_gaze_output_frequency(desired_hz)
            self.logger.info("Gaze output frequency set to %.0f Hz", desired_hz)
        else:
            closest = min(supported, key=lambda f: abs(f - desired_hz))
            self.eyetracker.set_gaze_output_frequency(closest)
            self.logger.warning(
                "%.0f Hz not supported by this device. Using %.0f Hz instead. Supported: %s",
                desired_hz,
                closest,
                sorted(supported),
            )

    def _gaze_callback(self, gaze_data: dict[str, Any]) -> None:
        """Enqueue each gaze sample from the tracker callback."""
        self.gaze_data = gaze_data
        self._gaze_queue.put(gaze_data)

    def start_tracking(self) -> None:
        """Subscribe to gaze data stream. Raises ValueError if not connected."""
        if not self.eyetracker:
            msg = "Eye tracker not connected"
            raise ValueError(msg)

        self.eyetracker.subscribe_to(
            tobii.EYETRACKER_GAZE_DATA, self._gaze_callback, as_dictionary=True
        )
        self.tracking = True
        self.logger.info("Gaze tracking started")

    def stop_tracking(self) -> None:
        """Unsubscribe from gaze data stream. No-op if not currently tracking."""
        if not self.eyetracker:
            msg = "Eye tracker not connected"
            raise ValueError(msg)

        if not self.tracking:
            return

        self.eyetracker.unsubscribe_from(
            tobii.EYETRACKER_GAZE_DATA, self._gaze_callback
        )
        self.tracking = False
        self.logger.info("Gaze tracking stopped")

    def set_display_area(
        self,
        width_mm: float | None = None,
        height_mm: float | None = None,
        mounting_offset_mm: float = 95.0,
        screen: int = 0,
    ) -> None:
        """
        Set the tracker's display area in the User Coordinate System (mm).

        Dimensions are auto-detected via xrandr/tkinter when omitted.
        mounting_offset_mm is the distance from the tracker's optical centre
        to the bottom screen edge (default 95 mm for standard under-bezel mounts).
        """
        if not self.eyetracker:
            msg = "Eye tracker not connected"
            raise ValueError(msg)

        if width_mm is None or height_mm is None:
            detected_w, detected_h = detect_screen_size_mm(screen)
            width_mm = width_mm if width_mm is not None else detected_w
            height_mm = height_mm if height_mm is not None else detected_h
            self.logger.info(
                "Auto-detected screen size: %.1f x %.1f mm", width_mm, height_mm
            )

        half_w = width_mm / 2.0
        top_y = mounting_offset_mm + height_mm
        display_area = tobii.DisplayArea(
            {
                "top_left": (-half_w, top_y, 0.0),
                "top_right": (half_w, top_y, 0.0),
                "bottom_left": (-half_w, mounting_offset_mm, 0.0),
            }
        )
        self.eyetracker.set_display_area(display_area)
        self.logger.info(
            "Display area set: %.1f x %.1f mm, offset=%.1f mm",
            width_mm,
            height_mm,
            mounting_offset_mm,
        )

    def calibrate(self, screen: int = 1, fullscreen: bool = True, **win_kwargs) -> None:
        """Run 5-point screen-based calibration. Uses self.win if set, otherwise creates a temporary window.

        After each calibration attempt, prompts the user to accept (SPACE) or redo (R).
        Loops until accepted.
        """
        # Use an externally supplied window, or create a temporary one.
        own_win = self.win is None
        if own_win:
            win = visual.Window(
                fullscr=fullscreen,
                screen=screen,
                units="pix",
                color="black",
                checkTiming=False,
                **win_kwargs,
            )
        else:
            win = self.win

        # Calibration target: outer ring shrinks toward a small inner dot to
        # pull the participant's fovea to the exact centre before sampling.
        outer_start = 40  # px - initial radius of outer ring
        outer_end = 6  # px - final radius after animation
        animate_dur = 3.0  # s  - shrink animation duration
        hold_dur = 0.5  # s  - static hold before collecting data
        inter_dur = 0.5  # s  - blank gap between points

        target_outer = visual.Circle(
            win, radius=outer_start, fillColor="white", lineColor="white", units="pix"
        )
        target_inner = visual.Circle(
            win, radius=4, fillColor="black", lineColor="black", units="pix"
        )

        prompt = visual.TextStim(
            win,
            text="Calibration complete.\n\nPress SPACE to accept  or  R to redo.",
            color="white",
            units="pix",
            height=24,
            wrapWidth=win.size[0] * 0.8,
        )

        calibration = tobii.ScreenBasedCalibration(self.eyetracker)

        while True:
            # Enter calibration mode.  If the display area has not been configured
            # on the device yet, attempt to derive it from the PsychoPy monitor
            # before retrying once.
            try:
                calibration.enter_calibration_mode()
            except tobii.EyeTrackerDisplayAreaNotValidError:
                self.set_display_area()
                calibration.enter_calibration_mode()

            self.logger.info(
                "Entered calibration mode for eye tracker %s",
                self.eyetracker.serial_number,
            )

            # Normalized (0-1) calibration point positions; (0,0) = top-left.
            # Points are kept 0.2 from each edge; 0.1 causes ~0.1 normalized-unit
            # accuracy loss on the Tobii Spark at extreme gaze angles.
            points_to_calibrate = [
                (0.5, 0.5),
                (0.2, 0.2),
                (0.2, 0.8),
                (0.8, 0.2),
                (0.8, 0.8),
            ]

            clock = core.Clock()
            for point in points_to_calibrate:
                escaped = show_calibration_point(
                    win,
                    calibration,
                    clock,
                    target_outer,
                    target_inner,
                    point,
                    outer_start,
                    outer_end,
                    animate_dur,
                    hold_dur,
                    inter_dur,
                )
                if escaped:
                    calibration.leave_calibration_mode()
                    if own_win:
                        win.close()
                    return

            # Clear the screen while computing.
            win.flip()

            calibration_result = calibration.compute_and_apply()
            print_calibration_results(calibration_result)
            self.logger.info(
                "Calibration result: %s, collected at %d points",
                calibration_result.status,
                len(calibration_result.calibration_points),
            )

            calibration.leave_calibration_mode()
            self.logger.info("Left calibration mode")

            # Prompt user to accept or redo.
            from psychopy import event as psycho_event

            prompt.draw()
            win.flip()
            keys = psycho_event.waitKeys(keyList=["space", "r"])
            if "space" in keys:
                self.logger.info("Calibration accepted by user")
                break
            self.logger.info("Calibration rejected — redoing")

        if own_win:
            win.close()

    def set_window(self, win: visual.Window) -> None:
        """Set the PsychoPy window used for calibration and validation."""
        self.win = win

    def get_trackbox_eye_pos(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ((left_x, left_y), (right_x, right_y)) in PsychoPy norm trackbox space, or (0.99, 0.99) if invalid."""
        if not self.gaze_data:
            return (0.99, 0.99), (0.99, 0.99)
        return get_trackbox_position(self.gaze_data)

    def get_avg_eye_distance(self) -> float:
        """Return average eye distance from tracker origin in centimeters, or 0 if no data."""
        if not self.gaze_data:
            return 0.0
        return get_eye_distance(get_3d_position(self.gaze_data))

    def get_avg_gaze_pos(self) -> tuple[float, float]:
        """Return average (x, y) gaze position in ADA normalized coords, or (nan, nan) if no data."""
        if not self.gaze_data:
            return (float("nan"), float("nan"))
        return get_gaze_position(self.gaze_data)
