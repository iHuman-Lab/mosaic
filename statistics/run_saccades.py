"""run_saccades.py – Run saccade detection on all subjects and LLM models.

Reads  data/{subject}/{llm_model}/eyetracker/eyetracker.h5
Saves  data/{subject}/{llm_model}/saccades/saccades.csv|parquet|h5
       data/saccades_summary.csv

Usage
-----
    python statistics/run_saccades.py
    python statistics/run_saccades.py --minlen 5 --maxvel 40 --maxacc 340
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "igaze"))

from igaze.detectors import saccade_detection  # noqa: E402


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _prepare_gaze(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract averaged x, y, time arrays from eyetracker DataFrame."""
    ch = cfg.get("streams", {}).get("eyetracker_channels", {})
    x_left  = ch.get("ch0", "left_gaze_x")
    y_left  = ch.get("ch1", "left_gaze_y")
    x_right = ch.get("ch2", "right_gaze_x")
    y_right = ch.get("ch3", "right_gaze_y")

    if x_right in df.columns and y_right in df.columns:
        x = ((df[x_left] + df[x_right]) / 2).to_numpy(dtype=float)
        y = ((df[y_left] + df[y_right]) / 2).to_numpy(dtype=float)
    else:
        x = df[x_left].to_numpy(dtype=float)
        y = df[y_left].to_numpy(dtype=float)

    time = df["timestamp"].to_numpy(dtype=float)
    # convert LSL timestamps (seconds since epoch) to relative ms
    if time.mean() > 1e6:
        time = (time - time[0]) * 1000.0

    return x, y, time


def _nearest_index(time: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(time - t)))


def run_saccades(config: dict, minlen: int, maxvel: int, maxacc: int, missing: float) -> None:
    data_root = ROOT / "data"
    subjects  = [s["id"] for s in config.get("subjects", [])]
    summaries = []

    for subject in subjects:
        subject_dir = data_root / subject
        if not subject_dir.exists():
            print(f"[SKIP] {subject}: no data folder found")
            continue

        llm_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]
        if not llm_dirs:
            print(f"[SKIP] {subject}: no LLM model folders found")
            continue

        for llm_dir in llm_dirs:
            llm_model = llm_dir.name
            h5_path   = llm_dir / "eyetracker" / "eyetracker.h5"

            if not h5_path.exists():
                print(f"[SKIP] {subject}/{llm_model}: eyetracker.h5 not found")
                continue

            print(f"[RUN]  {subject}/{llm_model} ...", end=" ", flush=True)

            df = pd.read_hdf(h5_path, key="eyetracker")
            x, y, time = _prepare_gaze(df, config)

            _, end_saccades = saccade_detection(
                x, y, time,
                missing=missing,
                minlen=minlen,
                maxvel=maxvel,
                maxacc=maxacc,
            )

            rows = []
            for sac_id, sac in enumerate(end_saccades, start=1):
                start_time, end_time, duration, x_start, y_start, x_end, y_end = sac
                amplitude = float(((x_end - x_start) ** 2 + (y_end - y_start) ** 2) ** 0.5)
                rows.append({
                    "subject":    subject,
                    "llm_model":  llm_model,
                    "saccade_id": sac_id,
                    "start_ms":   float(start_time),
                    "end_ms":     float(end_time),
                    "duration_ms": float(duration),
                    "x_start":    float(x_start),
                    "y_start":    float(y_start),
                    "x_end":      float(x_end),
                    "y_end":      float(y_end),
                    "amplitude":  amplitude,
                })

            n = len(rows)
            total_ms = sum(r["duration_ms"] for r in rows)
            mean_dur = total_ms / n if n > 0 else 0.0
            rate     = n / (total_ms / 1000.0) if total_ms > 0 else 0.0

            print(f"{n} saccades  mean_dur={mean_dur:.1f}ms  rate={rate:.2f}/s")

            # save per-subject/llm saccade events
            out_dir = llm_dir / "saccades"
            out_dir.mkdir(parents=True, exist_ok=True)

            if rows:
                sac_df = pd.DataFrame(rows)
                sac_df.to_csv(out_dir / "saccades.csv", index=False)
                sac_df.to_parquet(out_dir / "saccades.parquet", index=False)
                sac_df.to_hdf(out_dir / "saccades.h5", key="saccades", mode="w")

            summaries.append({
                "subject":              subject,
                "llm_model":            llm_model,
                "minlen_ms":            minlen,
                "maxvel":               maxvel,
                "maxacc":               maxacc,
                "n_saccades":           n,
                "total_duration_ms":    total_ms,
                "mean_duration_ms":     mean_dur,
                "mean_amplitude":       float(np.mean([r["amplitude"] for r in rows])) if rows else 0.0,
                "saccade_rate_per_sec": rate,
            })

    if summaries:
        summary_df = pd.DataFrame(summaries)
        out_path   = data_root / "saccades_summary.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"\nSummary saved to {out_path}")
        print(summary_df.to_string(index=False))
    else:
        print("\nNo saccades computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run saccade detection on all subjects.")
    parser.add_argument("--minlen",  type=int,   default=5,    help="Min saccade length in ms (default: 5)")
    parser.add_argument("--maxvel",  type=int,   default=40,   help="Max velocity threshold px/s (default: 40)")
    parser.add_argument("--maxacc",  type=int,   default=340,  help="Max acceleration threshold px (default: 340)")
    parser.add_argument("--missing", type=float, default=0.0,  help="Missing data value (default: 0.0)")
    parser.add_argument("--config",  type=Path,
                        default=Path(__file__).parent / "config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_saccades(config, minlen=args.minlen, maxvel=args.maxvel,
                 maxacc=args.maxacc, missing=args.missing)
