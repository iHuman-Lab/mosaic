
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf
import yaml

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_xdf(data_dir: str | Path, subject_id: str) -> Path | None:
    """Find an XDF file under data_dir whose name contains subject_id (case-insensitive)."""
    data_dir = Path(data_dir)
    for f in data_dir.rglob("*.xdf"):
        if subject_id.lower() in f.name.lower():
            return f
    return None


# ---------------------------------------------------------------------------
# Stream extraction
# ---------------------------------------------------------------------------

def _get_stream(streams: list[dict], name: str) -> dict | None:
    for s in streams:
        if s["info"].get("name", [""])[0] == name:
            return s
    return None


def _parse_game_stream(s: dict) -> pd.DataFrame:
    """Parse JSON game state samples into a flat DataFrame (drops array fields)."""
    rows = []
    for ts, v in zip(s["time_stamps"], s["time_series"]):
        try:
            sample = json.loads(v[0] if isinstance(v, (list, tuple)) else v)
        except (json.JSONDecodeError, TypeError):
            continue
        # keep only scalar fields (drop image, grid, etc.)
        flat = {"timestamp": float(ts)}
        for k, val in sample.items():
            if isinstance(val, (str, int, float, bool)) or val is None:
                flat[k] = val
        rows.append(flat)
    return pd.DataFrame(rows)


def _parse_eye_stream(s: dict, channel_map: dict | None = None) -> pd.DataFrame:
    """Parse eyetracker float samples into a DataFrame."""
    ts   = s["time_stamps"]
    data = np.array(s["time_series"])  # shape (N, channels)

    # try to get channel labels from metadata
    labels = None
    try:
        info    = s["info"]
        desc    = info.get("desc", [{}])
        if isinstance(desc, list):
            desc = desc[0] if desc else {}
        desc = desc or {}
        ch_node = desc.get("channels", [{}])
        if isinstance(ch_node, list):
            ch_node = ch_node[0] if ch_node else {}
        ch_node = ch_node or {}
        ch_list = ch_node.get("channel", [])
        if ch_list:
            labels = []
            for ch in ch_list:
                label = ch.get("label", ["?"])
                labels.append(label[0] if isinstance(label, list) else label)
    except Exception:
        labels = None

    if not labels:
        labels = [f"ch{i}" for i in range(data.shape[1])]

    df = pd.DataFrame(data, columns=labels)

    # apply channel name mapping from config if labels are generic
    if channel_map:
        df.rename(columns=channel_map, inplace=True)

    df.insert(0, "timestamp", ts)
    return df


# ---------------------------------------------------------------------------
# Split by trial_id
# ---------------------------------------------------------------------------

def _split_by_trial(game_df: pd.DataFrame,
                    eye_df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Split game and eyetracker data into per-trial segments using trial_id.
    Uses the time range of each trial block to slice the eyetracker.
    """
    if "trial_id" not in game_df.columns:
        raise ValueError(f"Field 'trial_id' not found in game stream. "
                         f"Available: {list(game_df.columns)}")

    result = {}
    for trial_id, group in game_df.groupby("trial_id", sort=False):
        t_start = group["timestamp"].iloc[0]
        t_end   = group["timestamp"].iloc[-1]

        eye_mask = (eye_df["timestamp"] >= t_start) & (eye_df["timestamp"] <= t_end)

        result[trial_id] = {
            "game":       group.reset_index(drop=True),
            "eyetracker": eye_df[eye_mask].reset_index(drop=True),
        }

    return result


# ---------------------------------------------------------------------------
# Per-subject loader
# ---------------------------------------------------------------------------

def load_subject(xdf_path: str | Path, config: dict) -> dict[str, dict[str, pd.DataFrame]]:
    """Load one XDF file and return data split by LLM model."""
    streams_cfg = config["streams"]
    game_name   = streams_cfg["game"]
    eye_name    = streams_cfg["eyetracker"]

    streams, _ = pyxdf.load_xdf(str(xdf_path))

    game_stream = _get_stream(streams, game_name)
    eye_stream  = _get_stream(streams, eye_name)

    if game_stream is None:
        raise RuntimeError(f"Game stream '{game_name}' not found in {xdf_path}")
    if eye_stream is None:
        raise RuntimeError(f"Eyetracker stream '{eye_name}' not found in {xdf_path}")

    channel_map = streams_cfg.get("eyetracker_channels", {})
    game_df = _parse_game_stream(game_stream)
    eye_df  = _parse_eye_stream(eye_stream, channel_map=channel_map)

    return _split_by_trial(game_df, eye_df)


# ---------------------------------------------------------------------------
# Load all subjects
# ---------------------------------------------------------------------------

def load_all(config: dict,
             data_dir: str | Path = "data",
             verbose: bool = True) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """
    Load all subjects listed in config.

    Returns:
        data[subject_id][llm_model]["game"]       -> DataFrame
        data[subject_id][llm_model]["eyetracker"] -> DataFrame
    """
    data_dir = Path(data_dir)
    subjects = config.get("subjects", [])
    result   = {}

    for entry in subjects:
        sid      = entry["id"] if isinstance(entry, dict) else entry
        xdf_path = find_xdf(data_dir, sid)

        if xdf_path is None:
            if verbose:
                print(f"[SKIP] {sid}: no XDF file found in {data_dir}")
            continue

        if verbose:
            print(f"[LOAD] {sid}: {xdf_path.name}")

        try:
            result[sid] = load_subject(xdf_path, config)
            if verbose:
                for model, streams in result[sid].items():
                    g = streams["game"]
                    e = streams["eyetracker"]
                    print(f"       {model:30s}  game={len(g):>6} rows  eye={len(e):>8} rows")
        except Exception as exc:
            if verbose:
                print(f"[ERROR] {sid}: {exc}")

    return result


# ---------------------------------------------------------------------------
# Save to disk
# ---------------------------------------------------------------------------

def save_all(data: dict, config: dict) -> None:
    """
    Save loaded data to disk as CSV, Parquet, and HDF5.
    Output path is built from config output_template:
        data/{subject}/{llm_model}/{stream}
    """
    template = config.get("xdf", {}).get("output_template", "data/{subject}/{llm_model}/{stream}")

    for sid, models in data.items():
        for model, streams in models.items():
            safe_model = model.replace("/", "-").replace(" ", "_")
            for key in ("game", "eyetracker"):
                df      = streams[key]
                out_dir = Path(template.format(subject=sid, llm_model=safe_model, stream=key))
                out_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_dir / f"{key}.csv", index=False)
                df.to_parquet(out_dir / f"{key}.parquet", index=False)
                df.to_hdf(out_dir / f"{key}.h5", key=key, mode="w")
                print(f"  saved  data/{sid}/{safe_model}/{key}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data = load_all(config, data_dir="data")

    print(f"\nSaving {len(data)} subject(s)...")
    save_all(data, config)
