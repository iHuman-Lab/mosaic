"""run_fixations.py – Run fixation detection on all subjects and LLM models.

Reads eyetracker.h5 from data/{subject}/{llm_model}/eyetracker/
Saves fixation results to  data/{subject}/{llm_model}/fixations/

Usage
-----
    python statistics/run_fixations.py
    python statistics/run_fixations.py --maxdist 25 --mindur 50
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

# allow imports from igaze/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "igaze"))

from igaze.fixation import extract_fixations_from_df, fixation_summary  # noqa: E402


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_fixations(config: dict, maxdist: int, mindur: int) -> None:
    data_root   = ROOT / "data"
    subjects    = [s["id"] for s in config.get("subjects", [])]
    output_tmpl = config.get("xdf", {}).get("output_template", "data/{subject}/{llm_model}/{stream}")

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

            Sfix, Efix = extract_fixations_from_df(df, maxdist=maxdist, mindur=mindur)
            summary    = fixation_summary(Efix)

            print(f"{summary['count']} fixations  "
                  f"mean_dur={summary['mean_duration_ms']:.1f}ms  "
                  f"rate={summary['fixation_rate_per_sec']:.2f}/s")

            # save fixation events
            out_dir = llm_dir / "fixations"
            out_dir.mkdir(parents=True, exist_ok=True)

            if Efix:
                fix_df = pd.DataFrame(
                    Efix, columns=["start_ms", "end_ms", "duration_ms", "x", "y"]
                )
                fix_df.to_csv(out_dir / "fixations.csv", index=False)
                fix_df.to_parquet(out_dir / "fixations.parquet", index=False)
                fix_df.to_hdf(out_dir / "fixations.h5", key="fixations", mode="w")

            # collect summary row
            summaries.append({
                "subject":              subject,
                "llm_model":            llm_model,
                "maxdist_px":           maxdist,
                "mindur_ms":            mindur,
                **summary,
            })

    if summaries:
        summary_df = pd.DataFrame(summaries)
        out_path   = data_root / "fixations_summary.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"\nSummary saved to {out_path}")
        print(summary_df.to_string(index=False))
    else:
        print("\nNo fixations computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fixation detection on all subjects.")
    parser.add_argument("--maxdist", type=int, default=25,
                        help="Max inter-sample distance in pixels (default: 25)")
    parser.add_argument("--mindur",  type=int, default=50,
                        help="Min fixation duration in ms (default: 50)")
    parser.add_argument("--config",  type=Path,
                        default=Path(__file__).parent / "config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_fixations(config, maxdist=args.maxdist, mindur=args.mindur)
