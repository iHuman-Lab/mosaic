import warnings
from contextlib import suppress
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
import yaml
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

COUNT_OUTCOMES = ["saved_victims", "step_count", "n_fixations", "n_saccades"]
CONTINUOUS_OUTCOMES = [
    "mean_fixation_duration",
    "total_fixation_time",
    "fixation_rate",
    "mean_saccade_duration",
    "total_saccade_time",
    "mean_amplitude",
    "saccade_rate",
]
DEFAULT_TRIAL_MERGE_KEYS = ["participant_id", "trial_id", "llm_provider", "prompt_type", "llm_model"]


def _load_glmm_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _resolve_config_path(config_path: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    resolved = Path(raw_path).expanduser()
    if resolved.is_absolute():
        return resolved
    return (config_path.parent / resolved).resolve()


def _load_csv_from_config(config_path: Path, files_cfg: dict, key: str, required: bool = False) -> pd.DataFrame:
    path = _resolve_config_path(config_path, files_cfg.get(key))
    if path is None:
        if required:
            raise ValueError(f"Missing required glmm.files.{key} in {config_path}")
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(f"Configured file for glmm.files.{key} does not exist: {path}")
    return pd.read_csv(path)


def _normalize_merge_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "subject_id" in df.columns and "participant_id" not in df.columns:
        df = df.rename(columns={"subject_id": "participant_id"})
    if "participant_id" in df.columns:
        df["participant_id"] = df["participant_id"].astype(str)
    return df


def _load_eye_metric_summaries(config_path: Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    files_cfg = config.get("glmm", {}).get("files", {})
    fixation_df = _normalize_merge_frame(_load_csv_from_config(config_path, files_cfg, "fixations_summary"))
    saccade_df = _normalize_merge_frame(_load_csv_from_config(config_path, files_cfg, "saccades_summary"))
    return fixation_df, saccade_df


def _load_trial_data(config_path: Path, config: dict) -> pd.DataFrame:
    files_cfg = config.get("glmm", {}).get("files", {})
    return _normalize_merge_frame(_load_csv_from_config(config_path, files_cfg, "trial_data", required=True))


def _subject_metadata(config: dict) -> pd.DataFrame:
    subjects = config.get("subjects", [])
    rows = []
    for subject in subjects:
        subject_id = subject.get("subject_id")
        if subject_id is None:
            return pd.DataFrame()
        rows.append(
            {
                "participant_id": str(subject_id),
                "expertise": subject.get("expertise", "unknown"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _apply_subject_metadata(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    metadata = _subject_metadata(config)
    if metadata.empty or "participant_id" not in df.columns:
        return df
    if "expertise" not in df.columns:
        return df.merge(metadata, on="participant_id", how="left")
    merged = df.merge(metadata, on="participant_id", how="left", suffixes=("", "_subject"))
    merged["expertise"] = merged["expertise"].fillna(merged.pop("expertise_subject"))
    return merged


def _merge_keys(config: dict, left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    configured = config.get("glmm", {}).get("merge_keys", DEFAULT_TRIAL_MERGE_KEYS)
    if not isinstance(configured, list):
        raise ValueError("glmm.merge_keys must be a list of column names")
    keys = [key for key in configured if key in left.columns and key in right.columns]
    if not keys:
        raise ValueError(
            "No configured merge keys were found in both dataframes. "
            f"Configured keys: {configured}"
        )
    return keys


def _normalize_model_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for target, source in [("participant_id", "subject_id"), ("llm_provider", "AI"), ("prompt_type", "Prompt")]:
        if target not in df.columns and source in df.columns:
            df[target] = df[source]
    mapping = {0: "sparse", 1: "detailed", "0": "sparse", "1": "detailed"}
    if "prompt_type" not in df.columns and "Detailed" in df.columns:
        df["prompt_type"] = df["Detailed"].map(mapping)
    if "llm_provider" not in df.columns and "Gemini" in df.columns:
        df["llm_provider"] = df["Gemini"].map({0: "other", 1: "gemini", "0": "other", "1": "gemini"})
    if "expertise" not in df.columns:
        df["expertise"] = "unknown"
    df["participant_id"] = df["participant_id"].astype(str)
    for col in ("llm_provider", "prompt_type"):
        df[col] = df[col].fillna("unknown").astype(str)
    df["expertise"] = df["expertise"].fillna("unknown").astype(str).str.strip().str.lower()
    valid = {"expert": "expert", "novice": "novice", "unknown": "unknown", "": "unknown"}
    df["expertise"] = df["expertise"].map(valid).fillna(df["expertise"])
    invalid = sorted(set(df["expertise"].dropna()) - {"expert", "novice", "unknown"})
    if invalid:
        raise ValueError(f"Invalid expertise value(s). Use only 'expert' or 'novice': {invalid}")
    return df


def prepare_glmm_df(config_path: str | Path) -> pd.DataFrame:
    config_path = Path(config_path)
    config = _load_glmm_config(config_path)
    df = _apply_subject_metadata(_normalize_model_df(_load_trial_data(config_path, config)), config)
    if "trial_id" in df.columns:
        df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce")
    for metrics in _load_eye_metric_summaries(config_path, config):
        if not metrics.empty:
            merge_keys = _merge_keys(config, df, metrics)
            df = df.merge(metrics, on=merge_keys, how="left")
    model_source = df["llm_model"] if "llm_model" in df.columns else df.get("llm_provider", "")
    prompt_source = df["prompt_type"] if "prompt_type" in df.columns else ""
    df["Gemini"] = pd.Series(model_source, index=df.index).astype(str).str.lower().str.contains("gemini").astype(int)
    df["Detailed"] = pd.Series(prompt_source, index=df.index).astype(str).str.lower().str.contains("detail").astype(int)
    for col in COUNT_OUTCOMES + CONTINUOUS_OUTCOMES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("saved_victims", "step_count"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return df


def _fixed_effects_formula(outcome: str, df: pd.DataFrame) -> str:
    return f"{outcome} ~ C(llm_provider) * C(prompt_type)" + (
        " + C(expertise)" if "expertise" in df.columns and df["expertise"].nunique(dropna=True) > 1 else ""
    )


def _fit_mixedlm(df: pd.DataFrame, outcome: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        return smf.mixedlm(_fixed_effects_formula(outcome, df), df, groups=df["participant_id"]).fit()


def _fit_poisson_glmm(df: pd.DataFrame, outcome: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        return PoissonBayesMixedGLM.from_formula(
            _fixed_effects_formula(outcome, df), {"participant": "0 + C(participant_id)"}, df
        ).fit_vb()


def _run_outcome_models(df: pd.DataFrame, outcomes: list[str], fit_fn) -> dict[str, object]:
    results = {}
    for outcome in outcomes:
        if outcome in df.columns and len(df.dropna(subset=[outcome])):
            with suppress(Exception):
                results[outcome] = fit_fn(df.dropna(subset=[outcome]).copy(), outcome)
    return results


def run_glmm_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _normalize_model_df(df)
    for col in COUNT_OUTCOMES + CONTINUOUS_OUTCOMES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["participant_id", "llm_provider", "prompt_type"]).copy()
    mixed = _run_outcome_models(df, CONTINUOUS_OUTCOMES, _fit_mixedlm)
    count = _run_outcome_models(df, COUNT_OUTCOMES, _fit_poisson_glmm)
    mixed_rows = [
        {"outcome": outcome, "term": term, "coef": model.params[term], "se": model.bse.get(term)}
        for outcome, model in mixed.items()
        for term in model.params.index
    ]
    count_rows = [
        {"outcome": outcome, "term": term, "coef": coef}
        for outcome, model in count.items()
        for term, coef in zip(model.model.exog_names, model.fe_mean)
    ]
    return pd.DataFrame(mixed_rows), pd.DataFrame(count_rows)
