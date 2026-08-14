from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import distance

# Constants
TRACKBOX_OUTSIDE = (0.99, 0.99)
DEFAULT_PUPIL = 0.0
TRACKBOX_SCALE = 1.7
MM_TO_CM = 10


def get_gaze_position(gaze_data: dict[str, Any]) -> tuple[float, float]:
    """Return average (x, y) gaze position in ADA normalized coords, or (nan, nan) if invalid."""
    left = gaze_data["left_gaze_point_on_display_area"]
    right = gaze_data["right_gaze_point_on_display_area"]
    x_vals = [v for v in (left[0], right[0]) if not np.isnan(v)]
    y_vals = [v for v in (left[1], right[1]) if not np.isnan(v)]

    if x_vals and y_vals:
        return float(np.mean(x_vals)), float(np.mean(y_vals))
    return np.nan, np.nan


def calculate_trackbox_coordinate(point: tuple, valid: int) -> tuple:
    """Convert a single eye's trackbox [0,1] coords to PsychoPy norm space, or TRACKBOX_OUTSIDE if invalid."""
    if valid != 1:
        return TRACKBOX_OUTSIDE
    x_norm = point[0] * 2 - 1
    y_norm = point[1] * 2 - 1
    return (-x_norm * TRACKBOX_SCALE, y_norm)


def get_trackbox_position(
    gaze_data: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ((left_x, left_y), (right_x, right_y)) eye positions in PsychoPy norm trackbox space."""
    left = gaze_data["left_gaze_origin_in_trackbox_coordinate_system"]
    right = gaze_data["right_gaze_origin_in_trackbox_coordinate_system"]

    left_pos = calculate_trackbox_coordinate(
        left, gaze_data["left_gaze_origin_validity"]
    )
    right_pos = calculate_trackbox_coordinate(
        right, gaze_data["right_gaze_origin_validity"]
    )
    return left_pos, right_pos


def get_3d_position(gaze_data: dict[str, Any]) -> tuple[float, float, float]:
    """Return average (x, y, z) eye origin in user coordinate system (mm), or (0, 0, 0) if invalid."""
    left = gaze_data["left_gaze_origin_in_user_coordinate_system"]
    right = gaze_data["right_gaze_origin_in_user_coordinate_system"]

    coordinates = [(left[i], right[i]) for i in range(3)]
    if not any(np.isnan(coord).all() for coord in coordinates):
        return tuple(np.nanmean(coord).item() for coord in coordinates)
    return (0, 0, 0)


def get_eye_distance(eye_pos: tuple[float, float, float]) -> float:
    """Return Euclidean distance of eye_pos from tracker origin in centimeters, or 0 if invalid."""
    if sum(eye_pos) > 0:
        return distance.euclidean(np.array(eye_pos) / MM_TO_CM, (0, 0, 0))
    return 0


def get_pupil_size(gaze_data: dict[str, Any]) -> float:
    """Return average pupil diameter in mm, or DEFAULT_PUPIL if either measurement is invalid."""
    left = gaze_data["left_pupil_diameter"]
    right = gaze_data["right_pupil_diameter"]

    if not (np.isnan(left) or np.isnan(right)):
        return float(np.mean([left, right]))
    return DEFAULT_PUPIL


def get_eye_validity(gaze_data: dict[str, Any]) -> int:
    """Return validity code: 0=none, 1=left only, 2=right only, 3=both."""
    left_valid = gaze_data["left_gaze_origin_validity"]
    right_valid = gaze_data["right_gaze_origin_validity"]

    return left_valid + (right_valid * 2)
