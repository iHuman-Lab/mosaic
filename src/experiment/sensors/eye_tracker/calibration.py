from __future__ import annotations

import logging

import numpy as np
import tobii_research as tobii
from psychopy import core, event, visual

from ixp.sensors.eye_tracker.utils import active_disp_to_psycho_pix

# Constants
COLORS = {
    "correct": [-1.0, 1.0, -1.0],  # Green
    "medium": [1.0, 1.0, 0.0],  # Yellow
    "wrong": [1.0, -1.0, -1.0],  # Red
    "white": [1.0, 1.0, 1.0],  # Default message color
    "black": [0.0, 0.0, 0.0],  # Default background
}

WINDOW_CONFIG = {
    "eye_radius": 0.07,
    "invalid_pos": 0.99,
    "text_height": 0.07,
    "unit": "norm",
    "line_width": 3,
    "max_gaze_positions": 6,
    "rect_scale": (1, 1),
}

DISTANCE_RANGES = {
    "correct": (55, 75),  # Correct range (min, max)
    "medium": [(45, 54), (76, 85)],  # Medium ranges (min, max) for both low and high
}


def validate_window(psycho_win: visual.Window) -> None:
    """Raise ValueError if psycho_win is None."""
    if psycho_win is None:
        msg = "No PsychoPy window available. Try calling run_trackbox() instead."
        raise ValueError(msg)


def create_visual_elements(psycho_win: visual.Window, rect_scale: tuple[float, float]):
    """Create all visual elements needed for eye tracking."""
    # Create viewing area
    eye_area = visual.Rect(
        psycho_win,
        fillColor=COLORS["black"],
        lineColor=COLORS["black"],
        pos=[0.0, 0.0],
        units=WINDOW_CONFIG["unit"],
        lineWidth=WINDOW_CONFIG["line_width"],
        width=rect_scale[0],
        height=rect_scale[1],
    )

    # Create eye stimuli
    left_eye = visual.Circle(
        psycho_win,
        fillColor=eye_area.fillColor,
        units=WINDOW_CONFIG["unit"],
        radius=WINDOW_CONFIG["eye_radius"],
    )
    right_eye = visual.Circle(
        psycho_win,
        fillColor=eye_area.fillColor,
        units=WINDOW_CONFIG["unit"],
        radius=WINDOW_CONFIG["eye_radius"],
    )

    # Create message display
    message = visual.TextStim(
        psycho_win,
        text=" ",
        color=COLORS["white"],
        units=WINDOW_CONFIG["unit"],
        pos=[0.0, -0.65],
        height=WINDOW_CONFIG["text_height"],
    )

    return eye_area, left_eye, right_eye, message


def get_eye_color(eye_distance: float) -> list[float]:
    """Return PsychoPy RGB color for eye stimuli based on viewing distance (cm)."""
    # Check for 'correct' distance range
    if DISTANCE_RANGES["correct"][0] <= eye_distance <= DISTANCE_RANGES["correct"][1]:
        return COLORS["correct"]

    # Check for 'medium' distance range (either of the two sub-ranges)
    for min_dist, max_dist in DISTANCE_RANGES["medium"]:
        if min_dist <= eye_distance <= max_dist:
            return COLORS["medium"]

    # Default to 'wrong' color
    return COLORS["wrong"]


def update_eye_stimuli(
    left_eye: visual.Circle, right_eye: visual.Circle, eye_distance: float, window_color
) -> None:
    """Color eye stimuli by distance; hide invalid eyes by blending with window_color."""
    color = get_eye_color(eye_distance)

    for eye in (left_eye, right_eye):
        eye.fillColor = color
        eye.lineColor = color
        if eye.pos[0] == WINDOW_CONFIG["invalid_pos"]:
            eye.fillColor = window_color
            eye.lineColor = window_color


def handle_user_input(psycho_win: visual.Window, stop_gaze_data) -> bool:
    """Return True if 'c' pressed, raise KeyboardInterrupt if 'q' pressed."""
    keys = event.getKeys(keyList=["q", "c"])
    if "q" in keys:
        cleanup(psycho_win, stop_gaze_data)
        msg = "Script aborted manually."
        raise KeyboardInterrupt(msg)
    if "c" in keys:
        logging.info("Proceeding to calibration.")
        stop_gaze_data()
        psycho_win.flip()
        return True
    return False


def cleanup(psycho_win: visual.Window, stop_gaze_data) -> None:
    """Stop tracking, close window, and quit PsychoPy."""
    stop_gaze_data()
    psycho_win.close()
    core.quit()


def draw_eye_positions(tracker, psycho_win: visual.Window):
    """Display real-time trackbox eye positions with distance feedback. Press 'c' to proceed or 'q' to abort."""
    validate_window(psycho_win)

    # Create visual elements
    eye_area, left_eye, right_eye, message = create_visual_elements(
        psycho_win, WINDOW_CONFIG["rect_scale"]
    )

    while True:
        # Update eye positions and distance
        left_eye.pos, right_eye.pos = tracker.get_trackbox_eye_pos()
        eye_distance = tracker.get_avg_eye_distance()

        # Update visual elements
        update_eye_stimuli(left_eye, right_eye, eye_distance, psycho_win.color)
        message.text = f"You're currently {int(eye_distance)} cm away from the screen.\n Press 'c' to calibrate or 'q' to abort."

        # Draw frame
        for element in (eye_area, left_eye, right_eye, message):
            element.draw()
        psycho_win.flip()

        # Handle user input
        if handle_user_input(psycho_win, tracker.stop_tracking):
            return
        event.clearEvents(eventType="keyboard")


DEFAULT_CALIBRATION_POINTS = [
    (0.1, 0.1),
    (0.9, 0.1),
    (0.5, 0.5),
    (0.1, 0.9),
    (0.9, 0.9),
]

CALIBRATION_POINT_RADII = [0.07, 0.05, 0.03, 0.02]
CALIBRATION_SHRINK_WAIT = 0.1
CALIBRATION_INTER_POINT_WAIT = 0.5


def run_calibration(
    tracker,
    psycho_win: visual.Window,
    calibration_points: list[tuple[float, float]] | None = None,
) -> tobii.CalibrationResult:
    """Run screen-based calibration with shrinking-circle targets. Returns the CalibrationResult."""
    if tracker.eyetracker is None:
        msg = "Eye tracker not connected."
        raise RuntimeError(msg)

    if calibration_points is None:
        calibration_points = DEFAULT_CALIBRATION_POINTS

    # Ensure the tracker window is set so ada2PsychoPix works
    tracker.set_window(psycho_win)

    calibration = tobii.ScreenBasedCalibration(tracker.eyetracker)
    calibration.enter_calibration_mode()
    logging.info("Entered calibration mode.")

    point_stim = visual.Circle(
        psycho_win,
        fillColor=COLORS["white"],
        lineColor=COLORS["white"],
        units="pix",
    )

    for point in calibration_points:
        draw_pos = active_disp_to_psycho_pix(point, psycho_win)

        # Shrink the circle to attract fixation
        for radius_norm in CALIBRATION_POINT_RADII:
            point_stim.radius = int(radius_norm * psycho_win.getSizePix()[1])
            point_stim.pos = draw_pos
            point_stim.draw()
            psycho_win.flip()
            core.wait(CALIBRATION_SHRINK_WAIT)

        result = calibration.collect_data(point[0], point[1])
        if result != tobii.CALIBRATION_STATUS_SUCCESS:
            logging.warning("Retrying calibration point %s.", point)
            calibration.collect_data(point[0], point[1])

        psycho_win.flip()  # blank between points
        core.wait(CALIBRATION_INTER_POINT_WAIT)

    calibration_result = calibration.compute_and_apply()
    calibration.leave_calibration_mode()
    logging.info("Calibration complete. Status: %s", calibration_result.status)

    return calibration_result


def get_default_validation_points() -> dict[str, tuple[float, float]]:
    """Return default 5-point ADA-normalized validation positions."""
    return {
        "1": (0.1, 0.1),
        "2": (0.9, 0.1),
        "3": (0.5, 0.5),
        "4": (0.1, 0.9),
        "5": (0.9, 0.9),
    }


def run_validation(tracker, point_dict: dict | None = None):
    """Show validation points and live gaze overlay to verify calibration accuracy."""
    if tracker.win is None:
        msg = "No experimental monitor specified. Try running set_monitor()."
        raise ValueError(msg)

    if point_dict is None:
        logging.info("Using 5-point default validation.")
        point_dict = get_default_validation_points()
    elif not isinstance(point_dict, dict):
        msg = "point_dict must be a dictionary with number keys and coordinate values."
        raise TypeError(msg)

    # Initialize validation
    tracker.start_tracking()
    core.wait(0.5)

    # Convert validation points
    point_positions = [
        active_disp_to_psycho_pix(pos, tracker.win) for pos in point_dict.values()
    ]

    # Create validation stimuli
    gaze_stim = visual.Circle(
        tracker.win,
        radius=WINDOW_CONFIG["eye_radius"],
        fillColor=COLORS["correct"],
        units=WINDOW_CONFIG["unit"],
    )

    point_stim = visual.Circle(
        tracker.win,
        radius=WINDOW_CONFIG["eye_radius"] / 2,
        fillColor=COLORS["white"],
        units=WINDOW_CONFIG["unit"],
    )

    message = visual.TextStim(
        tracker.win,
        text="Press 'q' to quit or 'c' to continue",
        color=COLORS["white"],
        pos=[0, -0.8],
        units=WINDOW_CONFIG["unit"],
    )

    # Initialize gaze position tracking
    gaze_positions = np.empty((0, 2))

    while True:
        # Update and smooth gaze positions
        gaze_positions = np.vstack(
            (gaze_positions, np.array(tracker.get_avg_gaze_pos()).reshape(1, 2))
        )
        if len(gaze_positions) > WINDOW_CONFIG["max_gaze_positions"]:
            gaze_positions = np.delete(gaze_positions, 0, axis=0)

        current_pos = np.nanmean(gaze_positions, axis=0)
        draw_pos = active_disp_to_psycho_pix(tuple(current_pos), tracker.win)

        # Draw current gaze position if valid
        if not np.isnan(draw_pos[0]):
            gaze_stim.pos = draw_pos
            gaze_stim.draw()

        # Draw validation points and message
        for pos in point_positions:
            point_stim.pos = pos
            point_stim.draw()
        message.draw()

        tracker.win.flip()

        # Handle user input
        if handle_user_input(tracker.win, tracker.stop_tracking):
            return

        event.clearEvents(eventType="keyboard")
