from __future__ import annotations

import logging
import re
import subprocess
import tkinter as tk

import numpy as np
import tobii_research as tobii
from psychopy import core, event, visual

_logger = logging.getLogger(__name__)
_XY_LEN = 2


def active_disp_to_psycho_pix(xy_coords, win):
    """Convert ADA normalized (x, y) to PsychoPy pixel coords (origin at centre, y-up)."""
    if not isinstance(xy_coords, tuple) or len(xy_coords) != _XY_LEN:
        msg = "XY coordinates must be a 2-tuple."
        raise ValueError(msg)

    if np.isnan(xy_coords[0]) or np.isnan(xy_coords[1]):
        return np.nan, np.nan

    mon_hw = win.size
    w_shift, h_shift = mon_hw[0] / 2, mon_hw[1] / 2

    return (
        int(xy_coords[0] * mon_hw[0] - w_shift),
        int((xy_coords[1] * mon_hw[1] - h_shift) * -1),
    )


def active_disp_to_mont_pix(xy_coords, win):
    """Convert ADA normalized (x, y) to monitor pixel coords (origin at top-left)."""
    if not isinstance(xy_coords, tuple) or len(xy_coords) != _XY_LEN:
        msg = "XY coordinates must be a 2-tuple."
        raise ValueError(msg)

    if np.isnan(xy_coords[0]) or np.isnan(xy_coords[1]):
        return np.nan, np.nan

    return (
        int(xy_coords[0] * win.getSizePix()[0]),
        int(xy_coords[1] * win.getSizePix()[1]),
    )


def detect_screen_size_mm(screen: int = 0) -> tuple[float, float]:
    """Return physical screen dimensions (width_mm, height_mm). Tries xrandr then tkinter."""
    try:
        output = subprocess.check_output(
            ["xrandr"],
            text=True,
            stderr=subprocess.DEVNULL,  # noqa: S607
        )
        connected = [line for line in output.splitlines() if " connected" in line]
        line = (
            connected[screen]
            if screen < len(connected)
            else (connected[0] if connected else "")
        )
        m = re.search(r"(\d+)mm x (\d+)mm", line)
        if m:
            return float(m.group(1)), float(m.group(2))
    except Exception:  # noqa: BLE001
        _logger.debug("xrandr screen size detection failed", exc_info=True)

    try:
        root = tk.Tk()
        root.withdraw()
        w_mm = root.winfo_screenmmwidth()
        h_mm = root.winfo_screenmmheight()
        root.destroy()
        if w_mm > 0 and h_mm > 0:
            return float(w_mm), float(h_mm)
    except Exception:  # noqa: BLE001
        _logger.debug("tkinter screen size detection failed", exc_info=True)

    msg = (
        "Could not auto-detect screen physical dimensions from the OS.  "
        "Call set_display_area(width_mm, height_mm) with your screen's "
        "physical dimensions manually."
    )
    raise RuntimeError(msg)


def show_calibration_point(  # noqa: PLR0913
    win: visual.Window,
    calibration: tobii.ScreenBasedCalibration,
    clock: core.Clock,
    target_outer: visual.Circle,
    target_inner: visual.Circle,
    point: tuple[float, float],
    outer_start: float,
    outer_end: float,
    animate_dur: float,
    hold_dur: float,
    inter_dur: float,
) -> bool:
    """Animate a shrinking target at point, collect gaze data. Returns True if Escape pressed."""
    pix = active_disp_to_psycho_pix(point, win)
    target_outer.pos = pix
    target_inner.pos = pix

    clock.reset()
    while clock.getTime() < animate_dur:
        t = clock.getTime() / animate_dur
        target_outer.radius = outer_start + (outer_end - outer_start) * t
        target_outer.draw()
        target_inner.draw()
        win.flip()
        if event.getKeys(["escape"]):
            calibration.leave_calibration_mode()
            return True

    target_outer.radius = outer_end
    target_outer.draw()
    target_inner.draw()
    win.flip()
    core.wait(hold_dur)

    _logger.debug("Collecting calibration data at %s", point)
    if calibration.collect_data(point[0], point[1]) != tobii.CALIBRATION_STATUS_SUCCESS:
        calibration.collect_data(point[0], point[1])

    win.flip()
    core.wait(inter_dur)
    return False


def print_calibration_results(result: tobii.CalibrationResult) -> None:
    """Print a per-point summary of calibration accuracy to stdout."""
    print(f"\n=== Calibration result: {result.status} ===")  # noqa: T201
    print(
        f"{'Point':>12}  {'L validity':>12}  {'L pos':>18}  {'R validity':>12}  {'R pos':>18}  {'Avg pos':>18}"
    )  # noqa: T201
    print("-" * 102)  # noqa: T201
    for cp in result.calibration_points:
        px, py = cp.position_on_display_area
        for sample in cp.calibration_samples:
            lv = sample.left_eye.validity
            rv = sample.right_eye.validity
            lx, ly = sample.left_eye.position_on_display_area
            rx, ry = sample.right_eye.position_on_display_area
            valid_xs = [v for v in (lx, rx) if not np.isnan(v)]
            valid_ys = [v for v in (ly, ry) if not np.isnan(v)]
            avg_x = sum(valid_xs) / len(valid_xs) if valid_xs else float("nan")
            avg_y = sum(valid_ys) / len(valid_ys) if valid_ys else float("nan")
            print(  # noqa: T201
                f"({px:.2f}, {py:.2f})  "
                f"{lv:>12}  ({lx:.4f}, {ly:.4f})  "
                f"{rv:>12}  ({rx:.4f}, {ry:.4f})  "
                f"({avg_x:.4f}, {avg_y:.4f})"
            )
    print("=" * 102 + "\n")  # noqa: T201
