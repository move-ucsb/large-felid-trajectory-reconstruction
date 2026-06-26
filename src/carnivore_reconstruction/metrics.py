"""Benchmark metrics for candidate and selected reconstructions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import ade, directness, discrete_frechet, discrete_dtw, path_length, rmse, spatial_rmse, EPS
from .tasks import ReconstructionTask
from .environment import environmental_error_dict


def evaluate_path(task: ReconstructionTask, path: np.ndarray, method: str, path_env: dict | None = None) -> dict:
    if task.truth_xy is None:
        raise ValueError("Evaluation requires a task with high-resolution truth_xy.")
    truth = task.truth_xy
    length = path_length(path)
    truth_length = path_length(truth)
    row = {
        "task_uid": task.task_uid,
        "method": method,
        "dataset": task.dataset,
        "taxon": task.taxon,
        "animal_id": task.animal_id,
        "setting_name": task.setting_name,
        "sex": task.sex,
        "age_class": task.age_class,
        "habitat_id": task.habitat_id,
        "study_system": task.study_system,
        "species_group": task.species_group,
        "ADE": ade(path, truth),
        "RMSE": rmse(path, truth),
        "spatial_RMSE": spatial_rmse(path, truth),
        "Frechet": discrete_frechet(path, truth),
        "DTW": discrete_dtw(path, truth),
        "path_length_m": length,
        "truth_path_length_m": truth_length,
        "path_length_log_error": abs(float(np.log((length + 1.0) / (truth_length + 1.0)))),
        "path_length_ratio": float(length / truth_length) if truth_length > EPS else np.nan,
        "path_length_ratio_error": abs(float(np.log(max(length, EPS) / max(truth_length, EPS)))) if truth_length > EPS else np.nan,
        "directness_error": abs(float(directness(path) - directness(truth))) if np.isfinite(directness(path)) and np.isfinite(directness(truth)) else np.nan,
        "within_50m": float(ade(path, truth) <= 50.0),
        "within_100m": float(ade(path, truth) <= 100.0),
    }
    if getattr(task, "truth_env", None) and path_env:
        row.update(environmental_error_dict(task.truth_env, path_env))
    return row


def add_linear_baseline_metrics(metrics: pd.DataFrame, baseline_method: str = "linear") -> pd.DataFrame:
    """Attach paired baseline-normalized metrics to a task-metric table.

    Raw ADE/RMSE in meters are still retained, but cross-setting comparisons are
    often fairer when every task is normalized by the same task's linear
    interpolation error.  Ratios below 1 and gain percentages above 0 indicate
    improvement over linear.
    """
    if metrics is None or metrics.empty or "task_uid" not in metrics.columns or "method" not in metrics.columns:
        return metrics
    out = metrics.copy()
    metric_map = {
        "ADE": "ade",
        "RMSE": "rmse",
        "spatial_RMSE": "spatial_rmse",
        "Frechet": "frechet",
        "DTW": "dtw",
        "path_length_log_error": "path_length_log_error",
        "directness_error": "directness_error",
    }
    available = [c for c in metric_map if c in out.columns]
    if not available:
        return out
    base = out[out["method"].astype(str).eq(str(baseline_method))][["task_uid"] + available].drop_duplicates("task_uid")
    if base.empty:
        return out
    rename = {c: f"linear_{c}" for c in available}
    base = base.rename(columns=rename)
    out = out.drop(columns=[f"linear_{c}" for c in available], errors="ignore").merge(base, on="task_uid", how="left")
    for raw_col in available:
        label = metric_map[raw_col]
        lin_col = f"linear_{raw_col}"
        vals = pd.to_numeric(out[raw_col], errors="coerce")
        lin = pd.to_numeric(out[lin_col], errors="coerce")
        ratio = vals / lin.replace(0, np.nan)
        out[f"{label}_ratio_to_linear"] = ratio.replace([np.inf, -np.inf], np.nan)
        out[f"{label}_gain_pct_vs_linear"] = 100.0 * (1.0 - out[f"{label}_ratio_to_linear"])
    if "ADE" in available:
        out["delta_ADE_vs_linear"] = pd.to_numeric(out["ADE"], errors="coerce") - pd.to_numeric(out["linear_ADE"], errors="coerce")
        out["better_than_linear"] = out["delta_ADE_vs_linear"].lt(0).astype(float)
    return out


def summarize_metrics(metrics: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or ["method"]
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    mm = add_linear_baseline_metrics(metrics)
    rows = []
    for key, g in mm.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        row.update({
            "n_tasks": int(g["task_uid"].nunique()),
            "ADE_mean": float(pd.to_numeric(g["ADE"], errors="coerce").mean()) if "ADE" in g.columns else np.nan,
            "ADE_median": float(pd.to_numeric(g["ADE"], errors="coerce").median()) if "ADE" in g.columns else np.nan,
            "RMSE_mean": float(pd.to_numeric(g["RMSE"], errors="coerce").mean()) if "RMSE" in g.columns else np.nan,
            "RMSE_median": float(pd.to_numeric(g["RMSE"], errors="coerce").median()) if "RMSE" in g.columns else np.nan,
            "Frechet_median": float(pd.to_numeric(g["Frechet"], errors="coerce").median()) if "Frechet" in g.columns else np.nan,
            "DTW_median": float(pd.to_numeric(g["DTW"], errors="coerce").median()) if "DTW" in g.columns else np.nan,
            "within_50m_rate": float(pd.to_numeric(g["within_50m"], errors="coerce").mean()) if "within_50m" in g.columns else np.nan,
            "within_100m_rate": float(pd.to_numeric(g["within_100m"], errors="coerce").mean()) if "within_100m" in g.columns else np.nan,
            "path_length_log_error_median": float(pd.to_numeric(g["path_length_log_error"], errors="coerce").median()) if "path_length_log_error" in g.columns else np.nan,
            "path_length_ratio_error_median": float(pd.to_numeric(g["path_length_ratio_error"], errors="coerce").median()) if "path_length_ratio_error" in g.columns else np.nan,
            "directness_error_median": float(pd.to_numeric(g["directness_error"], errors="coerce").median()) if "directness_error" in g.columns else np.nan,
            "better_than_linear_rate": float(pd.to_numeric(g["better_than_linear"], errors="coerce").mean()) if "better_than_linear" in g.columns else np.nan,
        })
        for col in [
            "ade_ratio_to_linear", "ade_gain_pct_vs_linear",
            "rmse_ratio_to_linear", "rmse_gain_pct_vs_linear",
            "frechet_ratio_to_linear", "frechet_gain_pct_vs_linear",
            "dtw_ratio_to_linear", "dtw_gain_pct_vs_linear",
        ]:
            if col in g.columns:
                row[f"{col}_median"] = float(pd.to_numeric(g[col], errors="coerce").median())
                row[f"{col}_mean"] = float(pd.to_numeric(g[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
