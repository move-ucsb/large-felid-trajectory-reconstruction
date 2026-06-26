"""Proposed motif-reconstruction method utilities.

This module implements the paper-facing proposed method in a clean,
package-friendly form:

* baselines are evaluated separately;
* the proposed candidate bank contains motif/time-geographic candidates plus
  internal direct/heading motifs used only as proposed-method fallbacks;
* Top-K representative paths are computed from the scored proposed bank;
* validation selects setting-level guarded methods and applies them to test.
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .candidates import generate_baseline_paths, generate_candidates_for_task
from .config import ReconstructionConfig
from .metrics import evaluate_path, summarize_metrics, add_linear_baseline_metrics
from .ranking import score_candidates
from .tasks import ReconstructionTask
from .timing import ProgressPrinter, status
from .environment import sample_path_environment


REPRESENTATIVE_BETAS = (0.25, 0.35, 0.50, 0.75)
REPRESENTATIVE_K = 20
# Representative families include both cost-rank scores and listwise probabilities.
REPRESENTATIVE_SCORE_COLS = ("cost_rank_score", "listwise_probability")
# Blend lambda is the weight assigned to the K20 weighted representative;
# the remaining weight is the direct/top1 anchor path.
BLEND_LAMBDAS = (0.25, 0.50, 0.75)
GUARD_QUANTILES = (0.50, 0.75)
MIN_VALIDATION_TASKS_PER_SETTING = 5
MIN_VALIDATION_GAIN_M = 0.25
MIN_VALIDATION_GAIN_PCT = 2.5
MAX_MEAN_WORSENING_M = 10.0
MIN_VALIDATION_BTL_RATE = 0.50

CANDIDATE_DIAGNOSTIC_COLUMNS = (
    "proposal_cost", "proposal_score", "cost_rank_score", "listwise_probability",
    "candidate_cost_context", "candidate_cost_direction", "candidate_cost_efficiency", "candidate_cost_source_shape",
    "candidate_cost_motif", "candidate_cost_step", "candidate_cost_turn", "candidate_cost_detour",
    "candidate_cost_directness", "candidate_cost_lateral", "candidate_cost_timegeo",
    "context_cost", "context_temporal_cost", "context_environment_cost", "context_demographic_cost",
    "context_n_environment_matches", "direction_cost", "efficiency_cost", "source_shape_cost",
    "source_path_ratio", "path_ratio", "expected_path_ratio", "target_path_ratio",
    "source_directness", "directness", "source_lateral_sign", "candidate_lateral_sign", "preferred_lateral_sign",
    "source_lateral_mean_norm", "source_lateral_abs_mean_norm", "step_violation_fraction", "step_q90_over_capacity",
)


def _task_raster_context(model, task: ReconstructionTask) -> tuple[dict, int | None]:
    """Return dataset-specific rasters and source EPSG for one task.

    Rasters should not be pooled across study systems because puma/Olympic
    tracks use EPSG:32610 while Thailand tracks use EPSG:32647. The model keeps
    both a dataset-specific mapping and a backward-compatible global mapping.
    """
    meta = getattr(model, "metadata", {}) or {}
    by_dataset = getattr(model, "environment_raster_paths_by_dataset", None) or meta.get("environment_raster_paths_by_dataset", {}) or {}
    epsg_by_dataset = getattr(model, "environment_epsg_by_dataset", None) or meta.get("environment_epsg_by_dataset", {}) or {}
    dataset = str(getattr(task, "dataset", ""))
    if dataset in by_dataset:
        paths = by_dataset.get(dataset) or {}
    else:
        paths = getattr(model, "environment_raster_paths", None) or meta.get("environment_raster_paths", {}) or {}
    epsg = epsg_by_dataset.get(dataset) if isinstance(epsg_by_dataset, dict) else None
    try:
        epsg = int(epsg) if epsg is not None and pd.notna(epsg) else None
    except Exception:
        epsg = None
    return dict(paths), epsg


@dataclass(frozen=True)
class MethodChoice:
    """One setting-level validation choice."""

    dataset: str
    taxon: str
    setting_name: str
    selected_method: str
    raw_best_method: str
    fallback_method: str
    validation_gain_median_m: float
    validation_mean_worsening_m: float
    validation_n_tasks: int
    guard_reason: str


def _num(s, default=np.nan) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _softmax_score(scores: np.ndarray, beta: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    scores = np.nan_to_num(scores, nan=-1e6, neginf=-1e6, posinf=0.0)
    z = float(beta) * (scores - np.nanmax(scores))
    w = np.exp(z)
    sw = float(np.nansum(w))
    if not np.isfinite(sw) or sw <= 0:
        return np.ones_like(w) / max(len(w), 1)
    return w / sw


def _candidate_path_map(candidate_paths: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {str(k): np.asarray(v, dtype=float) for k, v in candidate_paths.items()}


def _proposed_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    """Return only proposed candidates, never paper baselines."""
    if scored.empty:
        return scored.copy()
    if "method_family" in scored.columns:
        out = scored[scored["method_family"].astype(str).eq("proposed_motif")].copy()
        if not out.empty:
            return out
    # Backward-compatible fallback for older candidate tables.
    return scored[scored["origin"].astype(str).isin(["motif_timegeo_sr", "motif"])].copy()


def _top1_row(proposed: pd.DataFrame) -> pd.Series | None:
    if proposed.empty:
        return None
    score_col = "cost_rank_score" if "cost_rank_score" in proposed.columns else "proposal_score"
    x = proposed.copy()
    x["_score_tmp"] = _num(x[score_col], default=-1e6)
    return x.sort_values(["_score_tmp", "candidate_order"], ascending=[False, True], kind="mergesort").iloc[0]


def _source_relation_summary(x: pd.DataFrame) -> dict:
    """Summarize transfer-source labels for selected source candidates."""
    if x is None or x.empty:
        return {}
    rel_col = "source_transfer_relation" if "source_transfer_relation" in x.columns else None
    out = {}
    if rel_col is not None:
        rel = x[rel_col].fillna("unknown").astype(str)
        if len(rel):
            counts = rel.value_counts(dropna=False)
            out["top_source_transfer_relation"] = str(rel.iloc[0])
            out["dominant_source_transfer_relation"] = str(counts.index[0])
            out["source_transfer_relation_counts"] = ";".join(f"{k}:{int(v)}" for k, v in counts.items())
    for c in ["source_dataset", "source_taxon", "source_species_id", "source_habitat_id", "source_study_system", "source_animal_id"]:
        if c in x.columns and len(x):
            out[f"top_{c}"] = str(x[c].iloc[0])
    return out


def _candidate_diagnostics_from_row(row: pd.Series | dict, prefix: str = "") -> dict:
    """Return candidate-scoring diagnostics for metric/selection outputs."""
    out = {}
    getter = row.get if hasattr(row, "get") else lambda k, d=None: d
    for c in CANDIDATE_DIAGNOSTIC_COLUMNS:
        try:
            present = c in row
        except Exception:
            present = False
        if not present:
            continue
        v = getter(c, np.nan)
        key = f"{prefix}{c}" if prefix else c
        if isinstance(v, np.generic):
            v = v.item()
        out[key] = v
    return out


def _weighted_candidate_diagnostics(x: pd.DataFrame, weights: np.ndarray, prefix: str = "weighted_") -> dict:
    """Weighted average of diagnostic columns for representative paths."""
    out = {}
    if x is None or x.empty or weights is None or len(weights) == 0:
        return out
    n = min(len(x), len(weights))
    xx = x.iloc[:n].copy()
    ww = np.asarray(weights[:n], dtype=float)
    sw = float(np.nansum(ww))
    ww = ww / sw if np.isfinite(sw) and sw > 0 else np.ones(n, dtype=float) / max(n, 1)
    for c in CANDIDATE_DIAGNOSTIC_COLUMNS:
        if c not in xx.columns:
            continue
        vals = pd.to_numeric(xx[c], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(vals).any():
            out[f"{prefix}{c}"] = float(np.nansum(np.nan_to_num(vals, nan=0.0) * ww))
    # Mirror the key diagnostics without prefix for easier downstream plotting/QA.
    for c in [
        "candidate_cost_context", "candidate_cost_direction", "candidate_cost_efficiency", "candidate_cost_source_shape",
        "context_temporal_cost", "context_environment_cost", "context_demographic_cost", "context_n_environment_matches",
        "source_path_ratio", "path_ratio", "expected_path_ratio", "source_lateral_sign", "candidate_lateral_sign", "preferred_lateral_sign",
    ]:
        pc = f"{prefix}{c}"
        if pc in out:
            out[c] = out[pc]
    return out


def _retrieved_candidates(x: pd.DataFrame) -> pd.DataFrame:
    if x is None or x.empty or "style" not in x.columns:
        return pd.DataFrame(columns=x.columns if x is not None else [])
    return x[x["style"].astype(str).eq("retrieved_motif")].copy()


def _filter_shape_sources(x: pd.DataFrame, min_retrieved: int = 1) -> pd.DataFrame:
    """Prefer retrieved motifs over internal straight-line fallbacks when any exist."""
    if x is None or x.empty or "style" not in x.columns:
        return x
    retrieved = _retrieved_candidates(x)
    return retrieved if len(retrieved) >= int(min_retrieved) else x


def weighted_representative_path(
    proposed: pd.DataFrame,
    candidate_paths: dict[str, np.ndarray],
    score_col: str = "cost_rank_score",
    k: int = REPRESENTATIVE_K,
    beta: float = 0.50,
    shape_preserving: bool = True,
) -> tuple[np.ndarray | None, dict]:
    """Compute the K20 score-weighted representative path.

    This release defaults to a shape-preserving source set: if enough retrieved motifs
    exist, internal direct/heading fallbacks are excluded from the weighted
    average.  This avoids averaging a real motif bank with straight-line
    fallbacks, which visually collapses the representative path toward the
    endpoint chord.
    """
    if proposed.empty or score_col not in proposed.columns:
        return None, {}
    x = proposed.copy()
    if shape_preserving:
        x = _filter_shape_sources(x, min_retrieved=3)
    x["_score_tmp"] = _num(x[score_col], default=-1e6)
    x = x.sort_values(["_score_tmp", "candidate_order"], ascending=[False, True], kind="mergesort").head(int(k))
    path_map = _candidate_path_map(candidate_paths)
    arrays = []
    ids = []
    scores = []
    styles = []
    for _, row in x.iterrows():
        cid = str(row["candidate_id"])
        path = path_map.get(cid)
        if path is None or path.ndim != 2 or path.shape[1] != 2 or not np.isfinite(path).all():
            continue
        arrays.append(path)
        ids.append(cid)
        scores.append(float(row.get("_score_tmp", np.nan)))
        styles.append(str(row.get("style", row.get("origin", "unknown"))))
    if not arrays:
        return None, {}
    min_len = min(len(p) for p in arrays)
    arrays = [p[:min_len] for p in arrays if len(p) >= min_len]
    if not arrays:
        return None, {}
    W = _softmax_score(np.asarray(scores[: len(arrays)], dtype=float), beta)
    stack = np.stack(arrays, axis=0)
    xy = np.sum(stack * W[:, None, None], axis=0)
    dists = np.sqrt(((stack - xy[None, :, :]) ** 2).sum(axis=2)).mean(axis=1)
    top_style = styles[0] if styles else "unknown"
    source_summary = _source_relation_summary(x.head(len(arrays)))
    used_rows = x.head(len(arrays)).copy()
    styles_used = used_rows.get("style", pd.Series(dtype=str)).astype(str) if not used_rows.empty else pd.Series(dtype=str)
    meta = {
        "candidate_id": "+".join(ids[:5]),
        "candidate_origin": "weighted_topk_path",
        "score_col": score_col,
        "path_topK": int(k),
        "path_weight_beta": float(beta),
        "n_source_candidates": int(len(arrays)),
        "n_retrieved_source_candidates": int(styles_used.eq("retrieved_motif").sum()) if len(styles_used) else 0,
        "used_endpoint_fallback_only": int(not styles_used.eq("retrieved_motif").any()) if len(styles_used) else 1,
        "source_dispersion_mean_m": float(np.sum(dists * W)) if len(dists) else np.nan,
        "source_dispersion_max_m": float(np.nanmax(dists)) if len(dists) else np.nan,
        "source_score_margin": float(np.nanmax(scores) - np.nanmin(scores)) if len(scores) > 1 else np.inf,
        "top_source_style": top_style,
    }
    meta.update(source_summary)
    meta.update(_weighted_candidate_diagnostics(used_rows, W))
    return xy, meta


def _score_label(score_col: str) -> str:
    """Short stable label used in method names."""
    return "probability" if str(score_col) == "listwise_probability" else "cost_rank"


def _best_anchor_path(
    proposed: pd.DataFrame,
    candidate_paths: dict[str, np.ndarray],
    anchor: str = "top1",
    score_col: str = "cost_rank_score",
) -> tuple[np.ndarray | None, dict]:
    """Return an anchor path for blend variants.

    ``anchor='top1'`` uses the best scored proposed candidate. ``anchor='direct'``
    uses the best internal direct/heading-continuity proposed motif. These are
    the top-one and direct-like anchor rows.
    """
    if proposed.empty:
        return None, {}
    x = proposed.copy()
    if anchor == "top1":
        retrieved = _retrieved_candidates(x)
        if not retrieved.empty:
            x = retrieved.copy()
        else:
            return None, {}
    elif anchor == "direct":
        style = x.get("style", pd.Series("", index=x.index)).astype(str)
        direct_like = style.isin(["direct_identity", "heading_continuity", "fallback_linear"])
        if direct_like.any():
            x = x[direct_like].copy()
        else:
            return None, {}
    if score_col not in x.columns:
        score_col = "cost_rank_score" if "cost_rank_score" in x.columns else "proposal_score"
    x["_score_tmp"] = _num(x[score_col], default=-1e6)
    row = x.sort_values(["_score_tmp", "candidate_order"], ascending=[False, True], kind="mergesort").iloc[0]
    cid = str(row["candidate_id"])
    path = candidate_paths.get(cid)
    if path is None:
        return None, {}
    return np.asarray(path, dtype=float), {
        "anchor_candidate_id": cid,
        "anchor_style": str(row.get("style", "unknown")),
        "anchor_origin": str(row.get("origin", "motif_timegeo_sr")),
        "anchor_score_col": score_col,
    }


def _blend_paths(anchor_xy: np.ndarray, weighted_xy: np.ndarray, lambda_weighted: float) -> np.ndarray | None:
    """Blend an anchor path with a K20 weighted representative path."""
    if anchor_xy is None or weighted_xy is None:
        return None
    a = np.asarray(anchor_xy, dtype=float)
    w = np.asarray(weighted_xy, dtype=float)
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != 2 or w.shape[1] != 2:
        return None
    n = min(len(a), len(w))
    if n <= 0:
        return None
    lam = float(lambda_weighted)
    return (1.0 - lam) * a[:n] + lam * w[:n]


def proposed_paths_for_task(
    task: ReconstructionTask,
    scored: pd.DataFrame,
    candidate_paths: dict[str, np.ndarray],
    config: ReconstructionConfig,
    betas: Sequence[float] = REPRESENTATIVE_BETAS,
) -> dict[str, tuple[np.ndarray, dict]]:
    """Return proposed paths for one task.

    This implements the main representative families used by the proposed method: proposed top1 fallback, K20 weighted representatives using
    both deployable listwise score and listwise probability, and blend variants
    anchored to either the proposed top1 or a direct/heading proposed fallback.
    External paper baselines remain outside this candidate bank.
    """
    proposed = _proposed_candidates(scored)
    out: dict[str, tuple[np.ndarray, dict]] = {}

    # Top-1 is now strictly the best *retrieved* motif candidate. Endpoint
    # direct/heading candidates are reported separately as endpoint_fallback so
    # diagnostics cannot mistake a straight connector for a learned motif.
    retrieved = _retrieved_candidates(proposed)
    top = _top1_row(retrieved) if not retrieved.empty else None
    if top is not None:
        cid = str(top["candidate_id"])
        path = candidate_paths.get(cid)
        if path is not None:
            meta = {
                "candidate_id": cid,
                "candidate_origin": str(top.get("origin", "motif_timegeo_sr")),
                "style": str(top.get("style", "unknown")),
                "score_col": "cost_rank_score",
                "path_topK": 1,
                "path_weight_beta": np.nan,
                "n_source_candidates": 1,
                "n_retrieved_source_candidates": 1,
                "used_endpoint_fallback_only": 0,
                "source_dispersion_mean_m": 0.0,
                "source_method": "retrieved_motif_top1",
                **_source_relation_summary(pd.DataFrame([top])),
            }
            meta.update(_candidate_diagnostics_from_row(top))
            out["pretrained_motif_top1"] = (np.asarray(path, dtype=float), meta)

    # Diagnostic endpoint fallback, not a paper-facing proposed motif method.
    direct_anchor, direct_meta = _best_anchor_path(proposed, candidate_paths, anchor="direct", score_col="cost_rank_score")
    if direct_anchor is not None:
        meta = dict(direct_meta)
        meta.update({
            "candidate_id": meta.get("anchor_candidate_id", "endpoint_fallback"),
            "candidate_origin": "endpoint_fallback_path",
            "style": meta.get("anchor_style", "endpoint_fallback"),
            "score_col": "cost_rank_score",
            "path_topK": 1,
            "path_weight_beta": np.nan,
            "n_source_candidates": 1,
            "n_retrieved_source_candidates": 0,
            "used_endpoint_fallback_only": 1,
            "source_dispersion_mean_m": 0.0,
            "source_method": "endpoint_fallback",
            "top_source_transfer_relation": "proposed_internal",
            "dominant_source_transfer_relation": "proposed_internal",
        })
        out["endpoint_fallback"] = (np.asarray(direct_anchor, dtype=float), meta)

    for score_col in REPRESENTATIVE_SCORE_COLS:
        if score_col not in proposed.columns:
            continue
        label = _score_label(score_col)
        top_anchor, top_meta = _best_anchor_path(proposed, candidate_paths, anchor="top1", score_col=score_col)
        direct_anchor, direct_meta = _best_anchor_path(proposed, candidate_paths, anchor="direct", score_col=score_col)
        for beta in betas:
            weighted_method = f"pretrained_motif_weighted_{label}_K20_b{beta:g}"
            xy_w, meta_w = weighted_representative_path(proposed, candidate_paths, score_col=score_col, k=REPRESENTATIVE_K, beta=beta)
            if xy_w is None:
                continue
            meta_w = dict(meta_w)
            meta_w.update({
                "method_family_detail": "weighted_K20",
                "source_method": weighted_method,
                "score_col": score_col,
            })
            out[weighted_method] = (xy_w, meta_w)

            for lam in BLEND_LAMBDAS:
                if top_anchor is not None:
                    xy = _blend_paths(top_anchor, xy_w, lam)
                    if xy is not None:
                        method = f"pretrained_motif_blend_top1_weighted_{label}_K20_b{beta:g}_lam{lam:g}"
                        meta = dict(meta_w)
                        meta.update(top_meta)
                        meta.update({
                            "candidate_origin": "blend_top1_weighted_path",
                            "method_family_detail": "blend_top1_weighted",
                            "source_method": method,
                            "blend_lambda_weighted": float(lam),
                        })
                        out[method] = (xy, meta)
                if direct_anchor is not None:
                    xy = _blend_paths(direct_anchor, xy_w, lam)
                    if xy is not None:
                        method = f"pretrained_motif_blend_direct_weighted_{label}_K20_b{beta:g}_lam{lam:g}"
                        meta = dict(meta_w)
                        meta.update(direct_meta)
                        meta.update({
                            "candidate_origin": "blend_direct_weighted_path",
                            "method_family_detail": "blend_direct_weighted",
                            "source_method": method,
                            "blend_lambda_weighted": float(lam),
                        })
                        out[method] = (xy, meta)
    return out

def evaluate_proposed_methods_for_task(
    model,
    task: ReconstructionTask,
    include_baselines: bool = True,
    betas: Sequence[float] = REPRESENTATIVE_BETAS,
    keep_scored: bool = False,
) -> dict[str, object]:
    """Generate and evaluate all proposed-method variants for one truth-known task."""
    if task.truth_xy is None:
        raise ValueError("Proposed-method evaluation requires truth-known tasks.")
    t0 = time.perf_counter()
    metric_rows = []
    path_rows = []
    selected_rows = []
    save_paths = bool(getattr(model.config, "save_candidate_paths", False))
    eval_env = bool(getattr(model.config, "evaluate_environmental_exposure", False))
    raster_paths, raster_epsg = _task_raster_context(model, task) if eval_env else ({}, None)

    if include_baselines:
        for name, path in generate_baseline_paths(task).items():
            path_env = sample_path_environment(path, raster_paths, source_epsg=raster_epsg, raster_epsg=raster_epsg) if eval_env else {}
            metric_rows.append(evaluate_path(task, path, name, path_env=path_env))

    cand_table, cand_paths, cand_time = generate_candidates_for_task(task, model.library, model.config)
    scored, score_time = score_candidates(task, cand_table, cand_paths, model.movement_priors, model.config)
    proposed_paths = proposed_paths_for_task(task, scored, cand_paths, model.config, betas=betas)
    try:
        from .v31_direct_generators import generate_v31_paths_for_task
        proposed_paths.update(generate_v31_paths_for_task(model, task))
    except Exception as exc:
        status(f"Latent direct-generation paths skipped for task {getattr(task, 'task_uid', 'unknown')} after error: {exc}")

    try:
        from .ecological_candidates import generate_ecological_paths_for_task
        proposed_paths.update(generate_ecological_paths_for_task(model, task))
    except Exception as exc:
        status(f"Ecological candidate paths skipped for task {getattr(task, 'task_uid', 'unknown')} after error: {exc}")

    for method, (path, meta) in proposed_paths.items():
        path_env = sample_path_environment(path, raster_paths, source_epsg=raster_epsg, raster_epsg=raster_epsg) if eval_env else {}
        row = evaluate_path(task, path, method, path_env=path_env)
        row.update({k: v for k, v in meta.items() if k not in row})
        metric_rows.append(row)
        selected_rows.append({
            "task_uid": task.task_uid,
            "method": method,
            **meta,
        })
        if save_paths:
            for point_order, (x, y) in enumerate(path):
                path_rows.append({
                    "task_uid": task.task_uid,
                    "method": method,
                    "point_order": point_order,
                    "time": task.start_time + pd.to_timedelta(point_order * task.fine_dt_min, unit="min"),
                    "x": float(x),
                    "y": float(y),
                })

    runtime = {
        **cand_time,
        **score_time,
        "task_uid": task.task_uid,
        "total_seconds": time.perf_counter() - t0,
    }
    return {
        "metrics": pd.DataFrame(metric_rows),
        "selected": pd.DataFrame(selected_rows),
        "paths": pd.DataFrame(path_rows),
        "runtime": pd.DataFrame([runtime]),
        "scored": scored if keep_scored else pd.DataFrame(),
    }


def _auto_n_jobs(model, n_tasks: int) -> int:
    """Resolve thread count for validation/testing.

    Auto mode uses ``os.cpu_count() - 1`` as requested, with a lower bound of 1.
    Threads are used instead of processes to avoid repeatedly pickling the saved
    motif model on Windows/Jupyter.
    """
    cfg = getattr(model, "config", None)
    raw = getattr(cfg, "n_jobs", 0) if cfg is not None else 0
    try:
        raw_int = 0 if raw is None else int(raw)
    except Exception:
        raw_int = 0
    cpu_max = max(1, (os.cpu_count() or 2) - 1)
    if raw_int <= 0:
        n = cpu_max
    else:
        n = min(raw_int, cpu_max)
    return max(1, min(n, max(1, int(n_tasks))))


def _parallel_threshold(model) -> int:
    cfg = getattr(model, "config", None)
    try:
        return max(1, int(getattr(cfg, "parallel_threshold_tasks", 2)))
    except Exception:
        return 2


def evaluate_proposed_methods_for_tasks(
    model,
    tasks: Sequence[ReconstructionTask],
    include_baselines: bool = True,
    betas: Sequence[float] = REPRESENTATIVE_BETAS,
    keep_scored: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run proposed-method benchmark with per-task runtime.

    Uses a thread pool when ``model.config.n_jobs`` resolves to >1. This is
    useful for validation/testing notebooks and avoids copying the large model
    object into separate processes.
    """
    n_tasks = len(tasks)
    n_jobs = _auto_n_jobs(model, n_tasks)
    use_parallel = n_jobs > 1 and n_tasks >= _parallel_threshold(model)
    status(f"Running publication-ready benchmark on {n_tasks:,} task(s) with n_jobs={n_jobs}{' threads' if use_parallel else ' (serial)'}")
    metric_parts = []
    selected_parts = []
    path_parts = []
    runtime_parts = []
    scored_parts = []
    progress = ProgressPrinter("proposed benchmark", total=n_tasks, every=max(1, min(25, max(1, n_tasks//10))))

    def _run_one(task):
        return evaluate_proposed_methods_for_task(model, task, include_baselines=include_baselines, betas=betas, keep_scored=keep_scored)

    if use_parallel:
        ordered = [None] * n_tasks
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            future_to_i = {ex.submit(_run_one, task): i for i, task in enumerate(tasks)}
            done = 0
            for fut in as_completed(future_to_i):
                i = future_to_i[fut]
                task = tasks[i]
                ordered[i] = fut.result()
                done += 1
                progress.update(done, extra=f"latest={task.dataset}/{task.taxon}/{task.setting_name}")
        results = ordered
    else:
        results = []
        for i, task in enumerate(tasks, start=1):
            result = _run_one(task)
            results.append(result)
            progress.update(i, extra=f"latest={task.dataset}/{task.taxon}/{task.setting_name}")

    for result in results:
        if result is None:
            continue
        metric_parts.append(result["metrics"])
        selected_parts.append(result["selected"])
        path_parts.append(result["paths"])
        runtime_parts.append(result["runtime"])
        if keep_scored and result["scored"] is not None and not result["scored"].empty:
            scored_parts.append(result["scored"])

    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    guarded_metrics, choices = make_guarded_setting_selection(metrics)
    if not guarded_metrics.empty:
        metrics = pd.concat([metrics, guarded_metrics], ignore_index=True, sort=False)
    return {
        "task_metrics": metrics,
        "summary": summarize_metrics(metrics),
        "setting_summary": summarize_metrics(metrics, group_cols=["dataset", "taxon", "setting_name", "method"]),
        "selected_candidates": pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(),
        "paths": pd.concat(path_parts, ignore_index=True) if path_parts else pd.DataFrame(),
        "runtime": pd.concat(runtime_parts, ignore_index=True) if runtime_parts else pd.DataFrame(),
        "scored_candidates": pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame(),
        "setting_choices": choices,
    }


def _method_is_proposed_candidate(method: str) -> bool:
    m = str(method)
    return m.startswith("pretrained_motif_") and m != "pretrained_motif_guarded"


def _fallback_method(metrics: pd.DataFrame) -> str:
    """Stable full-coverage proposed fallback for guarded selection.

    Top-1 is no longer allowed to be an endpoint fallback, so for guarded
    full-coverage methods we prefer a representative K20 method. Endpoint
    fallback is diagnostic only and is not returned here.
    """
    vals = [m for m in metrics.get("method", pd.Series(dtype=str)).dropna().astype(str).unique() if _method_is_proposed_candidate(m)]
    preferred_exact = [
        "pretrained_motif_weighted_probability_K20_b0.25",
        "pretrained_motif_blend_top1_weighted_probability_K20_b0.25_lam0.25",
        "pretrained_motif_weighted_cost_rank_K20_b0.25",
        "pretrained_motif_top1",
    ]
    for m in preferred_exact:
        if m in vals:
            return m
    weighted = sorted([m for m in vals if "weighted" in m and "K20" in m])
    if weighted:
        return weighted[0]
    if vals:
        return sorted(vals)[0]
    raise ValueError("No proposed-method fallback found.")


def add_guarded_methods_from_validation(
    metrics: pd.DataFrame,
    validation_split: str = "validation",
    guard_qs: Sequence[float] = GUARD_QUANTILES,
) -> pd.DataFrame:
    """Create proposed-method coherence-guarded K20 methods from validation dispersion thresholds.

    The guard is metric-only: if a task's K20 source dispersion is above the
    validation threshold, the task falls back to the proposed top1 method.
    """
    if metrics.empty or "method" not in metrics.columns:
        return pd.DataFrame()
    fb_method = _fallback_method(metrics)
    fb = metrics[metrics["method"].eq(fb_method)].copy()
    if fb.empty:
        return pd.DataFrame()
    key_cols = ["task_uid"]
    fb_map = fb.set_index(key_cols, drop=False)
    raw_methods = [m for m in metrics["method"].dropna().astype(str).unique() if str(m).startswith("pretrained_motif_") and "K20" in str(m) and str(m) != fb_method]
    rows = []
    for method in raw_methods:
        sub = metrics[metrics["method"].eq(method)].copy()
        if sub.empty or "source_dispersion_mean_m" not in sub.columns:
            continue
        val = sub[sub.get("split", validation_split).eq(validation_split)] if "split" in sub.columns else sub.iloc[0:0]
        # If split is missing, infer from full metrics cannot happen in current package.
        if val.empty:
            continue
        disp = _num(val["source_dispersion_mean_m"], default=np.nan).replace(np.nan, np.nan)
        for q in guard_qs:
            if disp.notna().any():
                thr = float(np.nanquantile(disp.to_numpy(dtype=float), float(q)))
            else:
                thr = np.inf
            guarded_name = f"pretrained_motif_guarded_q{q:g}_{method}"
            for _, row in sub.iterrows():
                use_weighted = bool(float(row.get("source_dispersion_mean_m", np.inf)) <= thr)
                if use_weighted:
                    rr = row.copy()
                    rr["guard_used_weighted"] = 1
                    rr["guard_source_method"] = method
                else:
                    uid = row["task_uid"]
                    if uid not in fb_map.index:
                        continue
                    rr = fb_map.loc[uid].copy()
                    if isinstance(rr, pd.DataFrame):
                        rr = rr.iloc[0].copy()
                    rr["guard_used_weighted"] = 0
                    rr["guard_source_method"] = method
                rr["method"] = guarded_name
                rr["guard_dispersion_threshold_m"] = thr
                rows.append(rr.to_dict())
    return pd.DataFrame(rows)


def _attach_linear_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach paired linear deltas and normalized gains used by validation selection."""
    out = add_linear_baseline_metrics(metrics)
    if "linear_ADE" not in out.columns:
        linear = out[out["method"].eq("linear")][["task_uid", "ADE"]].rename(columns={"ADE": "linear_ADE"})
        out = out.drop(columns=["linear_ADE"], errors="ignore").merge(linear, on="task_uid", how="left")
    if "delta_ADE_vs_linear" not in out.columns:
        out["delta_ADE_vs_linear"] = pd.to_numeric(out["ADE"], errors="coerce") - pd.to_numeric(out["linear_ADE"], errors="coerce")
    if "better_than_linear" not in out.columns:
        out["better_than_linear"] = out["delta_ADE_vs_linear"].lt(0).astype(float)
    return out


def summarize_methods_for_selection(metrics: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Selection summary with method-selection columns."""
    if metrics.empty:
        return pd.DataFrame()
    mm = _attach_linear_delta(metrics)
    rows = []
    for key, g in mm.groupby(list(group_cols), dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        ade = pd.to_numeric(g["ADE"], errors="coerce")
        lin = pd.to_numeric(g["linear_ADE"], errors="coerce")
        row = dict(zip(group_cols, key))
        rmse = pd.to_numeric(g.get("RMSE", pd.Series(index=g.index, dtype=float)), errors="coerce")
        ple = pd.to_numeric(g.get("path_length_log_error", pd.Series(index=g.index, dtype=float)), errors="coerce")
        delta = ade - lin
        row.update({
            "n_tasks": int(g["task_uid"].nunique()),
            "ADE_median": float(ade.median()),
            "ADE_mean": float(ade.mean()),
            "RMSE_median": float(rmse.median()) if rmse.notna().any() else np.nan,
            "path_length_log_error_median": float(ple.median()) if ple.notna().any() else np.nan,
            "delta_ADE_median": float(delta.median()),
            "delta_ADE_mean": float(delta.mean()),
            "median_ADE_gain_m": float(-delta.median()) if delta.notna().any() else np.nan,
            "mean_ADE_gain_m": float(-delta.mean()) if delta.notna().any() else np.nan,
            "better_than_linear_rate": float((ade < lin).mean()) if lin.notna().any() else np.nan,
            "worse_than_linear_rate": float((ade >= lin).mean()) if lin.notna().any() else np.nan,
            "ADE_ratio_to_linear_median": float(pd.to_numeric(g.get("ade_ratio_to_linear", pd.Series(index=g.index, dtype=float)), errors="coerce").median()) if "ade_ratio_to_linear" in g.columns else np.nan,
            "ADE_gain_pct_vs_linear_median": float(pd.to_numeric(g.get("ade_gain_pct_vs_linear", pd.Series(index=g.index, dtype=float)), errors="coerce").median()) if "ade_gain_pct_vs_linear" in g.columns else np.nan,
            "RMSE_ratio_to_linear_median": float(pd.to_numeric(g.get("rmse_ratio_to_linear", pd.Series(index=g.index, dtype=float)), errors="coerce").median()) if "rmse_ratio_to_linear" in g.columns else np.nan,
            "Frechet_ratio_to_linear_median": float(pd.to_numeric(g.get("frechet_ratio_to_linear", pd.Series(index=g.index, dtype=float)), errors="coerce").median()) if "frechet_ratio_to_linear" in g.columns else np.nan,
            "DTW_ratio_to_linear_median": float(pd.to_numeric(g.get("dtw_ratio_to_linear", pd.Series(index=g.index, dtype=float)), errors="coerce").median()) if "dtw_ratio_to_linear" in g.columns else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _eligible_selection_summary(summary: pd.DataFrame, candidate_methods: set[str]) -> pd.DataFrame:
    """Filter a method summary to validation-eligible proposed methods."""
    if summary is None or summary.empty:
        return pd.DataFrame()
    g = summary[summary["method"].astype(str).isin(candidate_methods)].copy()
    if g.empty:
        return g
    g["median_ADE_gain_m"] = pd.to_numeric(g.get("median_ADE_gain_m", np.nan), errors="coerce")
    g["ADE_gain_pct_vs_linear_median"] = pd.to_numeric(g.get("ADE_gain_pct_vs_linear_median", np.nan), errors="coerce")
    g["ADE_ratio_to_linear_median"] = pd.to_numeric(g.get("ADE_ratio_to_linear_median", np.nan), errors="coerce")
    g["delta_ADE_mean"] = pd.to_numeric(g.get("delta_ADE_mean", np.nan), errors="coerce")
    g["better_than_linear_rate"] = pd.to_numeric(g.get("better_than_linear_rate", np.nan), errors="coerce")
    g["RMSE_median"] = pd.to_numeric(g.get("RMSE_median", np.nan), errors="coerce")
    g["path_length_log_error_median"] = pd.to_numeric(g.get("path_length_log_error_median", np.nan), errors="coerce")
    # Smaller is better. Prioritize normalized paired gain so hard/easy settings
    # are comparable, then win rate, then raw meters and shape preservation.
    g["selection_score"] = (
        -0.08 * g["ADE_gain_pct_vs_linear_median"].fillna(-1e6)
        -0.75 * g["better_than_linear_rate"].fillna(0.0)
        -0.02 * g["median_ADE_gain_m"].fillna(-1e6)
        +0.02 * g["RMSE_median"].fillna(1e6)
        +0.35 * g["path_length_log_error_median"].fillna(0.0)
    )
    return g


def _choose_method_from_summary(
    summary: pd.DataFrame,
    candidate_methods: set[str],
    min_gain_m: float,
    max_mean_worsening_m: float,
    min_btl_rate: float = MIN_VALIDATION_BTL_RATE,
    allow_best_if_no_eligible: bool = False,
) -> tuple[str | None, str | None, dict]:
    """Select a method by median gain, task-wise improvement rate, then RMSE/shape."""
    g = _eligible_selection_summary(summary, candidate_methods)
    if g.empty:
        return None, None, {"reason": "no_candidate_methods"}
    raw = g.sort_values(["selection_score", "ADE_median", "RMSE_median"], kind="mergesort").iloc[0]
    eligible = g[
        ((g["median_ADE_gain_m"].ge(float(min_gain_m))) | (g["ADE_gain_pct_vs_linear_median"].ge(float(MIN_VALIDATION_GAIN_PCT))))
        & (g["better_than_linear_rate"].ge(float(min_btl_rate)))
        & (g["delta_ADE_mean"].le(float(max_mean_worsening_m)))
    ].copy()
    if eligible.empty:
        if not allow_best_if_no_eligible:
            return None, str(raw["method"]), {
                "reason": "no_method_met_gain_btl_thresholds",
                "raw_best_method": str(raw["method"]),
                "raw_best_gain_median_m": float(raw.get("median_ADE_gain_m", np.nan)),
                "raw_best_gain_pct_median": float(raw.get("ADE_gain_pct_vs_linear_median", np.nan)),
                "raw_best_better_than_linear_rate": float(raw.get("better_than_linear_rate", np.nan)),
            }
        best = raw
        reason = "best_available_no_eligible"
    else:
        best = eligible.sort_values(["selection_score", "ADE_median", "RMSE_median"], kind="mergesort").iloc[0]
        reason = "selected"
    return str(best["method"]), str(raw["method"]), {
        "reason": reason,
        "validation_gain_median_m": float(best.get("median_ADE_gain_m", np.nan)),
        "validation_gain_pct_median": float(best.get("ADE_gain_pct_vs_linear_median", np.nan)),
        "validation_ADE_ratio_median": float(best.get("ADE_ratio_to_linear_median", np.nan)),
        "validation_mean_gain_m": float(best.get("mean_ADE_gain_m", np.nan)),
        "validation_mean_worsening_m": float(best.get("delta_ADE_mean", np.nan)),
        "validation_better_than_linear_rate": float(best.get("better_than_linear_rate", np.nan)),
        "validation_ADE_median": float(best.get("ADE_median", np.nan)),
        "validation_RMSE_median": float(best.get("RMSE_median", np.nan)),
        "validation_path_length_log_error_median": float(best.get("path_length_log_error_median", np.nan)),
        "validation_n_tasks": int(best.get("n_tasks", 0)),
    }



def _robust_global_candidate_methods(metrics: pd.DataFrame) -> set[str]:
    """Candidate methods eligible for the robust global paper method.

    We intentionally exclude setting-selected/guarded variants, Top-1, and
    endpoint fallback here because V20.1 testing showed those can overfit
    validation or sacrifice timestamp-wise accuracy.
    """
    vals = set(metrics.get("method", pd.Series(dtype=str)).dropna().astype(str).unique())
    out = set()
    for m in vals:
        if not _method_is_proposed_candidate(m):
            continue
        if m in {"pretrained_motif_top1", "endpoint_fallback", "pretrained_motif_guarded"}:
            continue
        if "guarded" in m:
            continue
        if "blend_direct_weighted" in m or ("weighted" in m and "K20" in m):
            out.add(m)
    return out


def make_robust_global_selection(
    metrics: pd.DataFrame,
    validation_split: str = "validation",
    min_btl_rate: float = 0.55,
    max_ade_ratio: float = 1.00,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a single validation-selected global paper method.

    This is deliberately more conservative than setting-level guarded selection:
    one method is chosen from validation and applied unchanged to all validation
    and test tasks. It reduces overfitting when setting-level validation counts
    are small.
    """
    if metrics.empty or "split" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    candidate_methods = _robust_global_candidate_methods(metrics)
    if not candidate_methods:
        return pd.DataFrame(), pd.DataFrame()
    val = metrics[metrics["split"].eq(validation_split)].copy()
    if val.empty:
        return pd.DataFrame(), pd.DataFrame()
    val_for_summary = val[val["method"].isin(candidate_methods | {"linear"})].copy()
    summary = summarize_methods_for_selection(val_for_summary, ["method"])
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = summary[summary["method"].isin(candidate_methods)].copy()
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    for c in ["better_than_linear_rate", "ADE_ratio_to_linear_median", "ADE_gain_pct_vs_linear_median", "RMSE_ratio_to_linear_median", "DTW_ratio_to_linear_median", "path_length_log_error_median"]:
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors="coerce")
    eligible = summary[
        summary["better_than_linear_rate"].ge(float(min_btl_rate))
        & summary["ADE_ratio_to_linear_median"].le(float(max_ade_ratio))
    ].copy()
    if eligible.empty:
        eligible = summary.copy()
        reason = "best_available_no_global_eligible"
    else:
        reason = "selected_global_robust"
    # Favor task-wise reliability first, then normalized timestamp error, then RMSE/DTW.
    sort_cols = ["better_than_linear_rate", "ADE_ratio_to_linear_median", "RMSE_ratio_to_linear_median", "DTW_ratio_to_linear_median", "path_length_log_error_median", "method"]
    eligible = eligible.sort_values(sort_cols, ascending=[False, True, True, True, True, True], kind="mergesort")
    chosen = str(eligible.iloc[0]["method"])
    source = metrics[metrics["method"].eq(chosen)].copy()
    if source.empty:
        return pd.DataFrame(), pd.DataFrame()
    out = source.copy()
    out["method"] = "pretrained_motif_robust_global"
    out["source_method"] = chosen
    choice = eligible.iloc[[0]].copy()
    choice["selected_method"] = chosen
    choice["paper_method"] = "pretrained_motif_robust_global"
    choice["selection_reason"] = reason
    return out, choice

def make_guarded_setting_selection(
    metrics: pd.DataFrame,
    validation_split: str = "validation",
    min_gain_m: float = MIN_VALIDATION_GAIN_M,
    max_mean_worsening_m: float = MAX_MEAN_WORSENING_M,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply validation-tuned setting-level selection with pooled fallbacks.

    Selection rule:
      1. Use setting-specific validation only when n >= MIN_VALIDATION_TASKS_PER_SETTING
         and a proposed method has median ADE gain >= min_gain_m and beats linear
         on at least 50% of validation tasks.
      2. If the setting is too small or has no eligible method, try pooled
         dataset/setting and dataset/taxon summaries.
      3. If still no eligible method, use the global validation-selected
         representative method. Endpoint fallback is diagnostic only.
    """
    if metrics.empty or "split" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()

    fb_method = _fallback_method(metrics)
    candidate_methods = {
        m for m in metrics["method"].dropna().astype(str).unique()
        if _method_is_proposed_candidate(m)
    }
    if not candidate_methods:
        return pd.DataFrame(), pd.DataFrame()

    val_all = metrics[metrics["split"].eq(validation_split)].copy()
    if val_all.empty:
        return pd.DataFrame(), pd.DataFrame()
    val_for_summary = val_all[val_all["method"].isin(candidate_methods | {"linear"})].copy()
    if val_for_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    setting_summary = summarize_methods_for_selection(val_for_summary, ["dataset", "taxon", "setting_name", "method"])
    dataset_setting_summary = summarize_methods_for_selection(val_for_summary, ["dataset", "setting_name", "method"])
    dataset_taxon_summary = summarize_methods_for_selection(val_for_summary, ["dataset", "taxon", "method"])
    global_summary = summarize_methods_for_selection(val_for_summary, ["method"])

    global_selected, global_raw, global_info = _choose_method_from_summary(
        global_summary, candidate_methods, min_gain_m, max_mean_worsening_m,
        allow_best_if_no_eligible=True,
    )
    if global_selected is None:
        global_selected = fb_method
        global_raw = fb_method
        global_info = {"reason": "fallback_method"}

    choices = []
    setting_keys = setting_summary[["dataset", "taxon", "setting_name"]].drop_duplicates()
    for _, kk in setting_keys.iterrows():
        dataset, taxon, setting = str(kk["dataset"]), str(kk["taxon"]), str(kk["setting_name"])
        g = setting_summary[
            setting_summary["dataset"].astype(str).eq(dataset)
            & setting_summary["taxon"].astype(str).eq(taxon)
            & setting_summary["setting_name"].astype(str).eq(setting)
        ].copy()
        n_tasks = int(g["n_tasks"].max()) if not g.empty else 0
        selected, raw_best, info = None, None, {"reason": "empty_setting", "validation_n_tasks": n_tasks}
        selection_level = "setting"
        if n_tasks >= MIN_VALIDATION_TASKS_PER_SETTING:
            selected, raw_best, info = _choose_method_from_summary(g, candidate_methods, min_gain_m, max_mean_worsening_m)
        else:
            info = {"reason": "too_few_validation_tasks", "validation_n_tasks": n_tasks}

        if selected is None:
            # Pool same dataset + sampling setting, useful for tiger/leopard 60→15.
            pg = dataset_setting_summary[
                dataset_setting_summary["dataset"].astype(str).eq(dataset)
                & dataset_setting_summary["setting_name"].astype(str).eq(setting)
            ].copy()
            pooled, pooled_raw, pooled_info = _choose_method_from_summary(pg, candidate_methods, min_gain_m, max_mean_worsening_m)
            if pooled is not None:
                selected, raw_best, info = pooled, pooled_raw, pooled_info
                selection_level = "dataset_setting_pool"

        if selected is None:
            # Pool same dataset + taxon across settings.
            pg = dataset_taxon_summary[
                dataset_taxon_summary["dataset"].astype(str).eq(dataset)
                & dataset_taxon_summary["taxon"].astype(str).eq(taxon)
            ].copy()
            pooled, pooled_raw, pooled_info = _choose_method_from_summary(pg, candidate_methods, min_gain_m, max_mean_worsening_m)
            if pooled is not None:
                selected, raw_best, info = pooled, pooled_raw, pooled_info
                selection_level = "dataset_taxon_pool"

        if selected is None:
            selected, raw_best, info = global_selected, global_raw, dict(global_info)
            selection_level = "global_validation"

        choices.append({
            "dataset": dataset,
            "taxon": taxon,
            "setting_name": setting,
            "selected_method": selected,
            "raw_best_method": raw_best or selected,
            "fallback_method": fb_method,
            "selection_level": selection_level,
            "validation_gain_median_m": info.get("validation_gain_median_m", info.get("raw_best_gain_median_m", np.nan)),
            "validation_mean_worsening_m": info.get("validation_mean_worsening_m", np.nan),
            "validation_better_than_linear_rate": info.get("validation_better_than_linear_rate", info.get("raw_best_better_than_linear_rate", np.nan)),
            "validation_ADE_median": info.get("validation_ADE_median", np.nan),
            "validation_RMSE_median": info.get("validation_RMSE_median", np.nan),
            "validation_path_length_log_error_median": info.get("validation_path_length_log_error_median", np.nan),
            "validation_n_tasks": int(info.get("validation_n_tasks", n_tasks) or n_tasks),
            "guard_reason": info.get("reason", "selected"),
        })
    choices_df = pd.DataFrame(choices)
    if choices_df.empty:
        return pd.DataFrame(), choices_df

    source = metrics[metrics["method"].isin(set(choices_df["selected_method"]) | {fb_method})].copy()
    source_map = {m: g.set_index("task_uid", drop=False) for m, g in source.groupby("method", sort=False)}
    choice_map = choices_df.set_index(["dataset", "taxon", "setting_name"])["selected_method"].to_dict()
    rows = []
    # Use fallback-method rows as coverage backbone. If a selected method is not
    # available for a task, fall back to the full-coverage representative method.
    base = metrics[metrics["method"].eq(fb_method)].copy()
    if base.empty:
        base = metrics[metrics["method"].astype(str).isin(candidate_methods)].drop_duplicates("task_uid").copy()
    for _, base_row in base.iterrows():
        choice_key = (base_row["dataset"], base_row["taxon"], base_row["setting_name"])
        selected_method = choice_map.get(choice_key, global_selected or fb_method)
        selected_map = source_map.get(selected_method)
        if selected_map is not None and not selected_map.empty and base_row["task_uid"] in selected_map.index:
            rr = selected_map.loc[base_row["task_uid"]]
            if isinstance(rr, pd.DataFrame):
                rr = rr.iloc[0]
            rr = rr.copy()
        else:
            rr = base_row.copy()
            selected_method = fb_method
        rr["method"] = "pretrained_motif_guarded"
        rr["source_method"] = selected_method
        rows.append(rr.to_dict())
    return pd.DataFrame(rows), choices_df

# -----------------------------------------------------------------------------
# V20.6 fast accuracy-tuning additions
# -----------------------------------------------------------------------------
# The additions below are deliberately lightweight. They keep the existing motif
# bank and baselines, but add: (1) phase-shifted motif variants, (2) confidence-
# adaptive motif/direct blends, and (3) a validation-tuned task-level selector
# that falls back to linear/heading paths when motif support is weak.

PHASE_GAMMAS = (0.75, 0.90, 1.10, 1.30)
DYNAMIC_BLEND_BETAS = (0.35, 0.50)
DYNAMIC_LAMBDA_MIN = 0.12
DYNAMIC_LAMBDA_MAX = 0.72


def _phase_warp_path(path: np.ndarray, gamma: float) -> np.ndarray | None:
    """Time-warp a path while preserving endpoints.

    gamma < 1 moves internal shape features earlier in normalized time;
    gamma > 1 moves them later. This targets the common failure mode where a
    motif contains a turn but places it at the wrong time along the gap.
    """
    p = np.asarray(path, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        return None
    n = len(p)
    t = np.linspace(0.0, 1.0, n)
    tq = np.clip(t ** float(gamma), 0.0, 1.0)
    x = np.interp(tq, t, p[:, 0])
    y = np.interp(tq, t, p[:, 1])
    out = np.column_stack([x, y])
    out[0] = p[0]
    out[-1] = p[-1]
    return out


def _candidate_confidence_from_rows(rows: pd.DataFrame) -> float:
    """A truth-free confidence score in [0, 1] for trusting motif geometry."""
    if rows is None or rows.empty:
        return 0.0
    x = rows.copy()
    # Use top rows only so low-quality tail candidates do not dominate.
    if "cost_rank_score" in x.columns:
        x["_score_tmp"] = _num(x["cost_rank_score"], default=-1e6)
        x = x.sort_values(["_score_tmp", "candidate_order"], ascending=[False, True], kind="mergesort").head(20)
    retrieved = _retrieved_candidates(x)
    n_retrieved = len(retrieved)
    support = min(1.0, n_retrieved / 8.0)
    top = x.iloc[0]

    def _cost_good(col, scale=1.0):
        if col not in top.index:
            return 0.5
        try:
            v = float(top.get(col))
        except Exception:
            return 0.5
        if not np.isfinite(v):
            return 0.5
        return float(np.exp(-max(v, 0.0) / float(scale)))

    direction = _cost_good("direction_cost", scale=1.0)
    context = _cost_good("context_cost", scale=1.2)
    efficiency = _cost_good("efficiency_cost", scale=1.0)
    source_shape = _cost_good("source_shape_cost", scale=1.0)

    # Score margin: high margin means top candidates agree more strongly.
    margin_score = 0.5
    if "cost_rank_score" in x.columns and len(x) > 1:
        s = pd.to_numeric(x["cost_rank_score"], errors="coerce").dropna().values
        if len(s) > 1:
            margin_score = float(np.clip((np.nanmax(s) - np.nanpercentile(s, 75)) / (abs(np.nanmax(s)) + 1e-6), 0, 1))

    # Penalize extreme path-ratio mismatch when available.
    ratio_score = 0.5
    for col in ["path_ratio", "source_path_ratio", "target_path_ratio"]:
        if col in top.index:
            try:
                r = float(top.get(col))
            except Exception:
                continue
            if np.isfinite(r) and r > 0:
                ratio_score = float(np.exp(-abs(np.log(max(r, 1e-6))) / 0.75))
                break

    conf = (
        0.22 * support
        + 0.18 * direction
        + 0.14 * context
        + 0.14 * efficiency
        + 0.12 * source_shape
        + 0.10 * margin_score
        + 0.10 * ratio_score
    )
    return float(np.clip(conf, 0.0, 1.0))


def _lambda_from_confidence(confidence: float) -> float:
    c = float(np.clip(confidence, 0.0, 1.0))
    return float(DYNAMIC_LAMBDA_MIN + (DYNAMIC_LAMBDA_MAX - DYNAMIC_LAMBDA_MIN) * c)


_previous_proposed_paths_for_task_v206 = proposed_paths_for_task


def proposed_paths_for_task(
    task: ReconstructionTask,
    scored: pd.DataFrame,
    candidate_paths: dict[str, np.ndarray],
    config: ReconstructionConfig,
    betas: Sequence[float] = REPRESENTATIVE_BETAS,
) -> dict[str, tuple[np.ndarray, dict]]:
    """V20.6 proposed paths: existing bank + phase and adaptive-blend variants."""
    out = _previous_proposed_paths_for_task_v206(task, scored, candidate_paths, config, betas=betas)
    proposed = _proposed_candidates(scored)
    retrieved = _retrieved_candidates(proposed)
    confidence = _candidate_confidence_from_rows(proposed)

    # Phase-shift top retrieved motif variants.
    top = _top1_row(retrieved) if not retrieved.empty else None
    if top is not None:
        cid = str(top.get("candidate_id", ""))
        base = candidate_paths.get(cid)
        if base is not None:
            for gamma in PHASE_GAMMAS:
                warped = _phase_warp_path(np.asarray(base, dtype=float), gamma)
                if warped is None:
                    continue
                method = f"pretrained_motif_top1_phase_g{gamma:g}"
                meta = {
                    "candidate_id": cid,
                    "candidate_origin": "phase_shifted_retrieved_motif",
                    "style": str(top.get("style", "retrieved_motif")),
                    "score_col": "cost_rank_score",
                    "path_topK": 1,
                    "path_weight_beta": np.nan,
                    "phase_gamma": float(gamma),
                    "selector_confidence": confidence,
                    "n_source_candidates": 1,
                    "n_retrieved_source_candidates": 1,
                    "used_endpoint_fallback_only": 0,
                    "source_method": method,
                    **_source_relation_summary(pd.DataFrame([top])),
                }
                meta.update(_candidate_diagnostics_from_row(top))
                out[method] = (warped, meta)

    # Confidence-adaptive direct/weighted blend. Low confidence shrinks toward
    # direct endpoint geometry; high confidence allows more motif shape.
    for score_col in REPRESENTATIVE_SCORE_COLS:
        if score_col not in proposed.columns:
            continue
        label = _score_label(score_col)
        direct_anchor, direct_meta = _best_anchor_path(proposed, candidate_paths, anchor="direct", score_col=score_col)
        if direct_anchor is None:
            continue
        for beta in DYNAMIC_BLEND_BETAS:
            weighted_xy, meta_w = weighted_representative_path(proposed, candidate_paths, score_col=score_col, k=REPRESENTATIVE_K, beta=beta, shape_preserving=True)
            if weighted_xy is None:
                continue
            lam = _lambda_from_confidence(confidence)
            xy = _blend_paths(direct_anchor, weighted_xy, lam)
            if xy is None:
                continue
            method = f"pretrained_motif_adaptive_blend_direct_weighted_{label}_K20_b{beta:g}"
            meta = dict(meta_w)
            meta.update(direct_meta)
            meta.update({
                "candidate_origin": "adaptive_confidence_blend_direct_weighted_path",
                "method_family_detail": "adaptive_blend_direct_weighted",
                "source_method": method,
                "blend_lambda_weighted": float(lam),
                "selector_confidence": confidence,
                "adaptive_confidence": confidence,
            })
            out[method] = (xy, meta)
    return out


def _adaptive_confidence_from_metric_row(row: pd.Series) -> float:
    vals = []
    for col, scale, sign in [
        ("selector_confidence", 1.0, 1),
        ("adaptive_confidence", 1.0, 1),
        ("direction_cost", 1.0, -1),
        ("context_cost", 1.2, -1),
        ("efficiency_cost", 1.0, -1),
        ("source_shape_cost", 1.0, -1),
    ]:
        if col not in row.index:
            continue
        try:
            v = float(row.get(col))
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        vals.append(v if sign > 0 else float(np.exp(-max(v, 0.0) / scale)))
    if "n_retrieved_source_candidates" in row.index:
        try:
            vals.append(min(1.0, float(row.get("n_retrieved_source_candidates")) / 8.0))
        except Exception:
            pass
    if not vals:
        return 0.5
    return float(np.clip(np.nanmean(vals), 0, 1))


def _adaptive_selector_candidate_methods(metrics: pd.DataFrame) -> list[str]:
    vals = list(metrics.get("method", pd.Series(dtype=str)).dropna().astype(str).unique())
    preferred = [
        "pretrained_motif_robust_global",
        "pretrained_motif_adaptive_blend_direct_weighted_cost_rank_K20_b0.35",
        "pretrained_motif_adaptive_blend_direct_weighted_cost_rank_K20_b0.5",
        "pretrained_motif_blend_direct_weighted_cost_rank_K20_b0.35_lam0.25",
        "pretrained_motif_blend_direct_weighted_cost_rank_K20_b0.5_lam0.25",
        "pretrained_motif_guarded",
    ]
    out = [m for m in preferred if m in vals]
    out.extend([m for m in vals if m.startswith("pretrained_motif_adaptive_blend") and m not in out])
    return out


def make_adaptive_task_selection(
    metrics: pd.DataFrame,
    validation_split: str = "validation",
    selector_name: str = "pretrained_motif_adaptive_selector",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation-tuned task-level selector between linear/heading and motif.

    The selector is intentionally simple and fast. It chooses one proposed method
    globally from validation, then tunes a confidence threshold: use the proposed
    path only when its truth-free support score exceeds the threshold; otherwise
    fall back to the better simple baseline selected on validation.
    """
    if metrics is None or metrics.empty or "task_uid" not in metrics.columns or "method" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    if "split" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()

    base_methods = [m for m in ["linear", "heading_hermite"] if m in set(metrics["method"].astype(str))]
    prop_methods = _adaptive_selector_candidate_methods(metrics)
    if not base_methods or not prop_methods:
        return pd.DataFrame(), pd.DataFrame()

    val = metrics[metrics["split"].astype(str).eq(validation_split)].copy()
    if val.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Pick baseline by validation median ADE ratio/ADE. Linear usually wins, but
    # heading-Hermite can win for some runs/settings.
    base_scores = []
    for bm in base_methods:
        g = val[val["method"].astype(str).eq(bm)]
        col = "ade_ratio_to_linear" if "ade_ratio_to_linear" in g.columns else "ADE"
        base_scores.append((float(pd.to_numeric(g[col], errors="coerce").median()), bm))
    base_scores = sorted(base_scores)
    fallback_method = base_scores[0][1]

    # Pick proposed method by validation win rate then median normalized error.
    summaries = summarize_methods_for_selection(val[val["method"].isin(set(prop_methods) | {"linear"})], ["method"])
    if summaries.empty:
        return pd.DataFrame(), pd.DataFrame()
    summaries = summaries[summaries["method"].isin(prop_methods)].copy()
    if summaries.empty:
        return pd.DataFrame(), pd.DataFrame()
    for c in ["better_than_linear_rate", "ADE_ratio_to_linear_median", "ADE_gain_pct_vs_linear_median"]:
        if c in summaries.columns:
            summaries[c] = pd.to_numeric(summaries[c], errors="coerce")
    summaries = summaries.sort_values(["better_than_linear_rate", "ADE_ratio_to_linear_median", "method"], ascending=[False, True, True], kind="mergesort")
    proposed_method = str(summaries.iloc[0]["method"])

    def build_for_threshold(thresh: float, source: pd.DataFrame) -> pd.DataFrame:
        rows = []
        prop = source[source["method"].astype(str).eq(proposed_method)].copy()
        fall = source[source["method"].astype(str).eq(fallback_method)].copy()
        prop = prop.drop_duplicates("task_uid")
        fall = fall.drop_duplicates("task_uid")
        pmap = {str(r["task_uid"]): r for _, r in prop.iterrows()}
        fmap = {str(r["task_uid"]): r for _, r in fall.iterrows()}
        for uid in sorted(set(pmap) | set(fmap)):
            prow = pmap.get(uid)
            frow = fmap.get(uid)
            if prow is None and frow is None:
                continue
            conf = _adaptive_confidence_from_metric_row(prow) if prow is not None else 0.0
            use_prop = (prow is not None) and (conf >= thresh)
            src = prow if use_prop else (frow if frow is not None else prow)
            row = src.copy()
            row["method"] = selector_name
            row["source_method"] = proposed_method if use_prop else fallback_method
            row["selector_confidence"] = conf
            row["selector_threshold"] = float(thresh)
            row["selector_used_proposed"] = int(use_prop)
            row["selector_fallback_method"] = fallback_method
            rows.append(row)
        return pd.DataFrame(rows)

    # Candidate thresholds: grid plus validation quantiles from proposed rows.
    vp = val[val["method"].astype(str).eq(proposed_method)].copy()
    confs = vp.apply(_adaptive_confidence_from_metric_row, axis=1) if not vp.empty else pd.Series(dtype=float)
    grid = list(np.linspace(0.0, 0.9, 10))
    if len(confs):
        grid += [float(x) for x in np.nanquantile(confs, [0.15, 0.25, 0.35, 0.50, 0.65, 0.75]) if np.isfinite(x)]
    grid = sorted(set(round(float(x), 4) for x in grid if np.isfinite(x)))

    scored = []
    for th in grid:
        cand = build_for_threshold(th, val)
        if cand.empty:
            continue
        cand = add_linear_baseline_metrics(pd.concat([val[val["method"].eq("linear")], cand], ignore_index=True, sort=False))
        sel = cand[cand["method"].eq(selector_name)].copy()
        if sel.empty:
            continue
        ade_ratio = pd.to_numeric(sel.get("ade_ratio_to_linear", pd.Series(dtype=float)), errors="coerce")
        ade = pd.to_numeric(sel.get("ADE", pd.Series(dtype=float)), errors="coerce")
        lin = pd.to_numeric(sel.get("linear_ADE", pd.Series(dtype=float)), errors="coerce")
        btl = float((ade < lin).mean()) if lin.notna().any() else np.nan
        med_ratio = float(ade_ratio.median()) if ade_ratio.notna().any() else np.inf
        med_ade = float(ade.median()) if ade.notna().any() else np.inf
        use_rate = float(pd.to_numeric(sel.get("selector_used_proposed", pd.Series(dtype=float)), errors="coerce").mean())
        scored.append((med_ratio, -btl if np.isfinite(btl) else 0.0, med_ade, th, use_rate))
    if not scored:
        return pd.DataFrame(), pd.DataFrame()
    scored = sorted(scored, key=lambda x: (x[0], x[1], x[2]))
    best_ratio, neg_btl, best_ade, best_thresh, use_rate = scored[0]

    out = build_for_threshold(best_thresh, metrics)
    choice = pd.DataFrame([{
        "paper_method": selector_name,
        "selected_method": proposed_method,
        "fallback_method": fallback_method,
        "selector_threshold": float(best_thresh),
        "validation_ADE_ratio_median": float(best_ratio),
        "validation_better_than_linear_rate": float(-neg_btl),
        "validation_ADE_median": float(best_ade),
        "validation_used_proposed_rate": float(use_rate),
        "selection_reason": "v20_6_confidence_adaptive_task_selector",
    }])
    return out, choice


# -----------------------------------------------------------------------------
# V20.7 CS-inspired residual-flow reconstruction and selector v2
# -----------------------------------------------------------------------------
# This release borrows lightweight ideas from video interpolation/restoration:
#   * predict a residual correction over a strong baseline rather than replacing
#     the whole path with a motif;
#   * use two-sided heading/Hermite flow as a baseline when previous/next fixes
#     exist;
#   * shrink motif residuals by a truth-free confidence score;
#   * use validation to select a conservative per-task fallback threshold.

RESIDUAL_BETAS = (0.35, 0.50)
RESIDUAL_LAMBDAS = (0.25, 0.50, 0.75)
RESIDUAL_CONFIDENCE_LAMBDAS = ("conf", "conf75")


def _linear_xy_between_endpoints(task: ReconstructionTask, n: int | None = None) -> np.ndarray:
    n = int(n or task.n_points)
    n = max(n, 2)
    t = np.linspace(0.0, 1.0, n)
    start = np.asarray(task.start_xy, dtype=float)
    end = np.asarray(task.end_xy, dtype=float)
    return start[None, :] + t[:, None] * (end - start)[None, :]


def _hermite_flow_xy(task: ReconstructionTask, n: int | None = None) -> np.ndarray:
    """Two-sided heading-flow / Hermite interpolation baseline.

    This is a trajectory analogue of bidirectional video interpolation: the start
    side is guided by the incoming heading and the end side by the outgoing
    heading when those adjacent coarse fixes exist.
    """
    n = int(n or task.n_points)
    n = max(n, 2)
    p0 = np.asarray(task.start_xy, dtype=float)
    p1 = np.asarray(task.end_xy, dtype=float)
    disp = float(np.linalg.norm(p1 - p0))
    if not np.isfinite(disp) or disp <= 0:
        return _linear_xy_between_endpoints(task, n=n)
    chord = p1 - p0
    # Default to chord tangent.  Scale by a conservative fraction of the gap.
    m0 = chord.copy()
    m1 = chord.copy()
    if getattr(task, "prev_xy", None) is not None:
        v = p0 - np.asarray(task.prev_xy, dtype=float)
        nv = float(np.linalg.norm(v))
        if np.isfinite(nv) and nv > 0:
            m0 = v / nv * disp
    if getattr(task, "next_xy", None) is not None:
        v = np.asarray(task.next_xy, dtype=float) - p1
        nv = float(np.linalg.norm(v))
        if np.isfinite(nv) and nv > 0:
            m1 = v / nv * disp
    # Avoid huge overshoot for sharp/uncertain headings.
    m0 = np.asarray(m0, dtype=float) * 0.55
    m1 = np.asarray(m1, dtype=float) * 0.55
    t = np.linspace(0.0, 1.0, n)
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    xy = h00[:, None]*p0 + h10[:, None]*m0 + h01[:, None]*p1 + h11[:, None]*m1
    xy[0] = p0; xy[-1] = p1
    return xy


def _resample_xy_to_n(path: np.ndarray, n: int) -> np.ndarray | None:
    p = np.asarray(path, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        return None
    n = int(n)
    if n <= 1:
        return None
    if len(p) == n:
        return p.copy()
    old = np.linspace(0.0, 1.0, len(p))
    new = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(new, old, p[:, 0]), np.interp(new, old, p[:, 1])])


def _motif_residual_path(task: ReconstructionTask, baseline_xy: np.ndarray, motif_xy: np.ndarray, lambda_residual: float) -> np.ndarray | None:
    """baseline + lambda * (motif - motif_linear), endpoints preserved."""
    base = np.asarray(baseline_xy, dtype=float)
    motif = _resample_xy_to_n(motif_xy, len(base))
    if motif is None or base.ndim != 2 or base.shape[1] != 2 or len(base) < 2:
        return None
    motif_linear = _linear_xy_between_endpoints(task, n=len(base))
    residual = motif - motif_linear
    lam = float(np.clip(lambda_residual, 0.0, 1.0))
    out = base + lam * residual
    out[0] = np.asarray(task.start_xy, dtype=float)
    out[-1] = np.asarray(task.end_xy, dtype=float)
    return out


def _confidence_lambda(confidence: float, mode: str | float) -> float:
    try:
        if isinstance(mode, str):
            c = float(np.clip(confidence, 0.0, 1.0))
            if mode == "conf75":
                return float(np.clip(0.75 * c, 0.05, 0.75))
            if mode == "conf":
                return float(np.clip(c, 0.05, 0.85))
        return float(np.clip(float(mode), 0.0, 1.0))
    except Exception:
        return 0.25


# Preserve the V20.6 path generator and extend it.
_previous_proposed_paths_for_task_v207 = proposed_paths_for_task


def proposed_paths_for_task(
    task: ReconstructionTask,
    scored: pd.DataFrame,
    candidate_paths: dict[str, np.ndarray],
    config: ReconstructionConfig,
    betas: Sequence[float] = REPRESENTATIVE_BETAS,
) -> dict[str, tuple[np.ndarray, dict]]:
    """V20.7 proposed paths: V20.6 bank + residual/flow variants."""
    out = _previous_proposed_paths_for_task_v207(task, scored, candidate_paths, config, betas=betas)
    proposed = _proposed_candidates(scored)
    if proposed is None or proposed.empty:
        return out
    confidence = _candidate_confidence_from_rows(proposed)
    linear_base = _linear_xy_between_endpoints(task, n=task.n_points)
    flow_base = _hermite_flow_xy(task, n=task.n_points)

    # Export the two-sided flow baseline as a diagnostic method inside the
    # proposed bank; external heading_hermite baseline remains separately scored.
    out["pretrained_motif_two_sided_flow"] = (flow_base, {
        "candidate_origin": "two_sided_heading_flow_baseline",
        "method_family_detail": "two_sided_flow",
        "source_method": "pretrained_motif_two_sided_flow",
        "selector_confidence": confidence,
        "adaptive_confidence": confidence,
        "n_source_candidates": 0,
        "n_retrieved_source_candidates": 0,
        "used_endpoint_fallback_only": 1,
        "top_source_transfer_relation": "proposed_internal",
        "dominant_source_transfer_relation": "proposed_internal",
    })

    # Use weighted representatives and phase-shifted top-1 motifs as residual
    # sources.  This is a trajectory equivalent of image/video residual
    # restoration: baseline timing is preserved; motif contributes curvature.
    residual_sources: list[tuple[str, np.ndarray, dict]] = []
    for score_col in REPRESENTATIVE_SCORE_COLS:
        if score_col not in proposed.columns:
            continue
        label = _score_label(score_col)
        for beta in RESIDUAL_BETAS:
            xy_w, meta_w = weighted_representative_path(proposed, candidate_paths, score_col=score_col, k=REPRESENTATIVE_K, beta=beta, shape_preserving=True)
            if xy_w is not None:
                residual_sources.append((f"weighted_{label}_K20_b{beta:g}", xy_w, dict(meta_w, score_col=score_col, path_weight_beta=float(beta))))
    retrieved = _retrieved_candidates(proposed)
    top = _top1_row(retrieved) if not retrieved.empty else None
    if top is not None:
        cid = str(top.get("candidate_id", ""))
        base = candidate_paths.get(cid)
        if base is not None:
            top_meta = {"candidate_id": cid, "score_col": "cost_rank_score", **_source_relation_summary(pd.DataFrame([top]))}
            top_meta.update(_candidate_diagnostics_from_row(top))
            residual_sources.append(("top1", np.asarray(base, dtype=float), dict(top_meta)))
            for gamma in PHASE_GAMMAS:
                warped = _phase_warp_path(np.asarray(base, dtype=float), gamma)
                if warped is not None:
                    residual_sources.append((f"top1_phase_g{gamma:g}", warped, dict(top_meta, phase_gamma=float(gamma))))

    seen = set()
    for source_label, motif_xy, source_meta in residual_sources:
        if motif_xy is None:
            continue
        # Avoid too many nearly duplicate sources.
        key = (source_label, len(motif_xy))
        if key in seen:
            continue
        seen.add(key)
        for lam_mode in list(RESIDUAL_LAMBDAS) + list(RESIDUAL_CONFIDENCE_LAMBDAS):
            lam = _confidence_lambda(confidence, lam_mode)
            for baseline_name, baseline_xy, short in [
                ("linear", linear_base, "linear"),
                ("flow", flow_base, "flow"),
                ("hermite", flow_base, "hermite"),
            ]:
                xy = _motif_residual_path(task, baseline_xy, motif_xy, lambda_residual=lam)
                if xy is None:
                    continue
                lam_label = str(lam_mode).replace(".", "p")
                method = f"pretrained_motif_residual_{short}_{source_label}_lam{lam_label}"
                meta = dict(source_meta)
                meta.update({
                    "candidate_origin": "baseline_plus_motif_residual",
                    "method_family_detail": f"residual_{short}",
                    "source_method": method,
                    "residual_source": source_label,
                    "residual_baseline": baseline_name,
                    "residual_lambda": float(lam),
                    "selector_confidence": confidence,
                    "adaptive_confidence": confidence,
                    "n_source_candidates": int(source_meta.get("n_source_candidates", 1) or 1),
                    "n_retrieved_source_candidates": int(source_meta.get("n_retrieved_source_candidates", 1) or 1),
                    "used_endpoint_fallback_only": int(source_meta.get("used_endpoint_fallback_only", 0) or 0),
                })
                out[method] = (xy, meta)
    return out


def _adaptive_selector_candidate_methods(metrics: pd.DataFrame) -> list[str]:
    """V20.7 candidate method order for selector v2.

    Exclude the old guarded selector from adaptive tuning because V20.1/V20.6
    showed it can overfit small validation groups.  Residual-flow methods are
    preferred when they exist, then robust global/adaptive blends.
    """
    vals = set(metrics.get("method", pd.Series(dtype=str)).dropna().astype(str))
    preferred = [
        "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.35_lamconf",
        "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.5_lamconf",
        "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.35_lam0p5",
        "pretrained_motif_residual_hermite_weighted_cost_rank_K20_b0.35_lamconf",
        "pretrained_motif_residual_linear_weighted_cost_rank_K20_b0.35_lamconf",
        "pretrained_motif_robust_global",
        "pretrained_motif_adaptive_blend_direct_weighted_cost_rank_K20_b0.35",
        "pretrained_motif_adaptive_blend_direct_weighted_cost_rank_K20_b0.5",
    ]
    out = [m for m in preferred if m in vals]
    residuals = sorted([m for m in vals if m.startswith("pretrained_motif_residual_") and m not in out])
    out.extend(residuals)
    blends = sorted([m for m in vals if m.startswith("pretrained_motif_adaptive_blend") and m not in out])
    out.extend(blends)
    if "pretrained_motif_robust_global" in vals and "pretrained_motif_robust_global" not in out:
        out.append("pretrained_motif_robust_global")
    return out


def _row_confidence_v207(row: pd.Series | None) -> float:
    if row is None:
        return 0.0
    vals = []
    for col in ["selector_confidence", "adaptive_confidence"]:
        if col in row.index:
            try:
                v = float(row.get(col))
                if np.isfinite(v):
                    vals.append(np.clip(v, 0, 1))
            except Exception:
                pass
    for col, scale in [
        ("direction_cost", 1.0),
        ("candidate_cost_direction", 1.0),
        ("context_cost", 1.2),
        ("candidate_cost_context", 1.2),
        ("efficiency_cost", 1.0),
        ("candidate_cost_efficiency", 1.0),
        ("source_shape_cost", 1.0),
        ("candidate_cost_source_shape", 1.0),
    ]:
        if col in row.index:
            try:
                v = float(row.get(col))
                if np.isfinite(v):
                    vals.append(float(np.exp(-max(v, 0) / scale)))
            except Exception:
                pass
    if "n_retrieved_source_candidates" in row.index:
        try:
            vals.append(float(np.clip(float(row.get("n_retrieved_source_candidates")) / 8.0, 0, 1)))
        except Exception:
            pass
    if not vals:
        return 0.5
    return float(np.clip(np.nanmean(vals), 0, 1))


def _selector_score_frame(df: pd.DataFrame, selector_name: str) -> dict:
    sel = df[df["method"].astype(str).eq(selector_name)].copy()
    if sel.empty:
        return {"med_ratio": np.inf, "btl": -np.inf, "mean_ratio": np.inf, "med_ade": np.inf}
    ade_ratio = pd.to_numeric(sel.get("ade_ratio_to_linear", pd.Series(dtype=float)), errors="coerce")
    ade = pd.to_numeric(sel.get("ADE", pd.Series(dtype=float)), errors="coerce")
    lin = pd.to_numeric(sel.get("linear_ADE", pd.Series(dtype=float)), errors="coerce")
    mean_ratio = float(ade_ratio.mean()) if ade_ratio.notna().any() else np.inf
    med_ratio = float(ade_ratio.median()) if ade_ratio.notna().any() else np.inf
    btl = float((ade < lin).mean()) if lin.notna().any() else -np.inf
    med_ade = float(ade.median()) if ade.notna().any() else np.inf
    return {"med_ratio": med_ratio, "btl": btl, "mean_ratio": mean_ratio, "med_ade": med_ade}


def make_adaptive_task_selection(
    metrics: pd.DataFrame,
    validation_split: str = "validation",
    selector_name: str = "pretrained_motif_adaptive_selector_v2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V20.7 conservative adaptive selector.

    Chooses among linear/heading-Hermite and residual/proposed methods.  The
    selector never uses task-level truth at test time; it only uses a validation-
    selected proposed method plus a confidence threshold.  Candidate methods and
    thresholds are tuned on validation with paired ADE/linear ratio as primary
    objective and win rate as tie-breaker.
    """
    if metrics is None or metrics.empty or "task_uid" not in metrics.columns or "method" not in metrics.columns or "split" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    all_methods = set(metrics["method"].dropna().astype(str))
    base_methods = [m for m in ["linear", "heading_hermite", "pretrained_motif_two_sided_flow"] if m in all_methods]
    prop_methods = _adaptive_selector_candidate_methods(metrics)
    if not base_methods or not prop_methods:
        return pd.DataFrame(), pd.DataFrame()
    val = metrics[metrics["split"].astype(str).eq(validation_split)].copy()
    if val.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Baseline fallback: choose by median normalized ADE on validation.
    base_scores = []
    for bm in base_methods:
        g = val[val["method"].astype(str).eq(bm)].copy()
        if g.empty:
            continue
        col = "ade_ratio_to_linear" if "ade_ratio_to_linear" in g.columns else "ADE"
        base_scores.append((float(pd.to_numeric(g[col], errors="coerce").median()), bm))
    if not base_scores:
        return pd.DataFrame(), pd.DataFrame()
    fallback_method = sorted(base_scores)[0][1]

    def build_selector(source: pd.DataFrame, proposed_method: str, threshold: float) -> pd.DataFrame:
        prop = source[source["method"].astype(str).eq(proposed_method)].drop_duplicates("task_uid").copy()
        fall = source[source["method"].astype(str).eq(fallback_method)].drop_duplicates("task_uid").copy()
        if prop.empty and fall.empty:
            return pd.DataFrame()
        pmap = {str(r["task_uid"]): r for _, r in prop.iterrows()}
        fmap = {str(r["task_uid"]): r for _, r in fall.iterrows()}
        rows = []
        for uid in sorted(set(pmap) | set(fmap)):
            prow = pmap.get(uid)
            frow = fmap.get(uid)
            conf = _row_confidence_v207(prow)
            use_prop = (prow is not None) and (conf >= float(threshold))
            src = prow if use_prop else (frow if frow is not None else prow)
            if src is None:
                continue
            row = src.copy()
            row["method"] = selector_name
            row["source_method"] = proposed_method if use_prop else fallback_method
            row["selector_confidence"] = conf
            row["selector_threshold"] = float(threshold)
            row["selector_used_proposed"] = int(use_prop)
            row["selector_fallback_method"] = fallback_method
            row["selector_version"] = "v20.7_residual_flow_selector"
            rows.append(row)
        return pd.DataFrame(rows)

    # Evaluate several proposed methods and thresholds on validation.
    scored = []
    threshold_grid = list(np.linspace(0.0, 0.95, 20))
    for pm in prop_methods:
        vp = val[val["method"].astype(str).eq(pm)].copy()
        if vp.empty:
            continue
        confs = vp.apply(_row_confidence_v207, axis=1)
        if len(confs):
            threshold_grid_pm = threshold_grid + [float(x) for x in np.nanquantile(confs, [0.10, 0.25, 0.40, 0.50, 0.65, 0.75, 0.90]) if np.isfinite(x)]
        else:
            threshold_grid_pm = threshold_grid
        for th in sorted(set(round(float(x), 4) for x in threshold_grid_pm if np.isfinite(x))):
            cand = build_selector(val, pm, th)
            if cand.empty:
                continue
            paired = add_linear_baseline_metrics(pd.concat([val[val["method"].astype(str).eq("linear")], cand], ignore_index=True, sort=False))
            score = _selector_score_frame(paired, selector_name)
            use_rate = float(pd.to_numeric(cand.get("selector_used_proposed", pd.Series(dtype=float)), errors="coerce").mean())
            # Penalize selectors that collapse to all-proposed unless they are clearly strong.
            collapse_penalty = 0.004 if use_rate >= 0.98 else 0.0
            scored.append({
                "proposed_method": pm,
                "threshold": float(th),
                "med_ratio": score["med_ratio"] + collapse_penalty,
                "raw_med_ratio": score["med_ratio"],
                "btl": score["btl"],
                "mean_ratio": score["mean_ratio"],
                "med_ade": score["med_ade"],
                "use_rate": use_rate,
            })
    if not scored:
        return pd.DataFrame(), pd.DataFrame()
    score_df = pd.DataFrame(scored)
    score_df = score_df.sort_values(["med_ratio", "mean_ratio", "btl", "med_ade", "use_rate"], ascending=[True, True, False, True, True], kind="mergesort")
    best = score_df.iloc[0]
    out = build_selector(metrics, str(best["proposed_method"]), float(best["threshold"]))
    choice = pd.DataFrame([{
        "paper_method": selector_name,
        "selected_method": str(best["proposed_method"]),
        "fallback_method": fallback_method,
        "selector_threshold": float(best["threshold"]),
        "validation_ADE_ratio_median": float(best["raw_med_ratio"]),
        "validation_better_than_linear_rate": float(best["btl"]),
        "validation_ADE_median": float(best["med_ade"]),
        "validation_used_proposed_rate": float(best["use_rate"]),
        "selection_reason": "v20_7_residual_flow_adaptive_selector_v2",
    }])
    return out, choice


# V20.7 final lightweight override: keep residual experiment fast by exporting a
# compact residual-flow grid rather than every possible source/lambda/baseline.
def proposed_paths_for_task(
    task: ReconstructionTask,
    scored: pd.DataFrame,
    candidate_paths: dict[str, np.ndarray],
    config: ReconstructionConfig,
    betas: Sequence[float] = REPRESENTATIVE_BETAS,
) -> dict[str, tuple[np.ndarray, dict]]:
    """V20.7 lightweight paths: V20.6 bank + compact residual-flow variants."""
    out = _previous_proposed_paths_for_task_v207(task, scored, candidate_paths, config, betas=betas)
    proposed = _proposed_candidates(scored)
    if proposed is None or proposed.empty:
        return out
    confidence = _candidate_confidence_from_rows(proposed)
    linear_base = _linear_xy_between_endpoints(task, n=task.n_points)
    flow_base = _hermite_flow_xy(task, n=task.n_points)
    out["pretrained_motif_two_sided_flow"] = (flow_base, {
        "candidate_origin": "two_sided_heading_flow_baseline",
        "method_family_detail": "two_sided_flow",
        "source_method": "pretrained_motif_two_sided_flow",
        "selector_confidence": confidence,
        "adaptive_confidence": confidence,
        "n_source_candidates": 0,
        "n_retrieved_source_candidates": 0,
        "used_endpoint_fallback_only": 1,
        "top_source_transfer_relation": "proposed_internal",
        "dominant_source_transfer_relation": "proposed_internal",
    })

    residual_sources: list[tuple[str, np.ndarray, dict]] = []
    # Main weighted source: cost-rank is most deployable/stable from V20.1/V20.6.
    if "cost_rank_score" in proposed.columns:
        for beta in (0.35, 0.50):
            xy_w, meta_w = weighted_representative_path(proposed, candidate_paths, score_col="cost_rank_score", k=REPRESENTATIVE_K, beta=beta, shape_preserving=True)
            if xy_w is not None:
                residual_sources.append((f"weighted_cost_rank_K20_b{beta:g}", xy_w, dict(meta_w, score_col="cost_rank_score", path_weight_beta=float(beta))))
    # Top1 and two phase variants for turn-location testing.
    retrieved = _retrieved_candidates(proposed)
    top = _top1_row(retrieved) if not retrieved.empty else None
    if top is not None:
        cid = str(top.get("candidate_id", ""))
        base = candidate_paths.get(cid)
        if base is not None:
            top_meta = {"candidate_id": cid, "score_col": "cost_rank_score", **_source_relation_summary(pd.DataFrame([top]))}
            top_meta.update(_candidate_diagnostics_from_row(top))
            residual_sources.append(("top1", np.asarray(base, dtype=float), dict(top_meta)))
            for gamma in (0.90, 1.10):
                warped = _phase_warp_path(np.asarray(base, dtype=float), gamma)
                if warped is not None:
                    residual_sources.append((f"top1_phase_g{gamma:g}", warped, dict(top_meta, phase_gamma=float(gamma))))

    # Compact method grid: residual over linear and two-sided flow, with either
    # confidence-scaled lambda or fixed 0.5.  This is enough to test the idea
    # without making validation/testing much slower.
    for source_label, motif_xy, source_meta in residual_sources:
        for lam_mode in ("conf", 0.50):
            lam = _confidence_lambda(confidence, lam_mode)
            for baseline_name, baseline_xy, short in [
                ("linear", linear_base, "linear"),
                ("flow", flow_base, "flow"),
            ]:
                xy = _motif_residual_path(task, baseline_xy, motif_xy, lambda_residual=lam)
                if xy is None:
                    continue
                lam_label = str(lam_mode).replace(".", "p")
                method = f"pretrained_motif_residual_{short}_{source_label}_lam{lam_label}"
                meta = dict(source_meta)
                meta.update({
                    "candidate_origin": "baseline_plus_motif_residual",
                    "method_family_detail": f"residual_{short}",
                    "source_method": method,
                    "residual_source": source_label,
                    "residual_baseline": baseline_name,
                    "residual_lambda": float(lam),
                    "selector_confidence": confidence,
                    "adaptive_confidence": confidence,
                    "n_source_candidates": int(source_meta.get("n_source_candidates", 1) or 1),
                    "n_retrieved_source_candidates": int(source_meta.get("n_retrieved_source_candidates", 1) or 1),
                    "used_endpoint_fallback_only": int(source_meta.get("used_endpoint_fallback_only", 0) or 0),
                })
                out[method] = (xy, meta)
    return out


# -----------------------------------------------------------------------------
# V20.8 oracle-distilled selector
# -----------------------------------------------------------------------------
# V20.7 showed that the candidate pool can approach oracle, but hand-written
# scoring does not choose the near-oracle candidate reliably.  This selector is
# trained only on validation rows, using deployable method/task/candidate
# diagnostics, and then predicts a task-specific method for validation/test rows.
# It does not use test truth or test ADE to choose a method; ADE is only used
# later by the benchmark to evaluate the selected row.

ORACLE_DISTILLED_SELECTOR_NAME = "pretrained_motif_oracle_distilled_selector"


def _oracle_distilled_candidate_methods(metrics: pd.DataFrame) -> list[str]:
    """Compact method pool for the learned selector.

    We keep baselines and a small set of stable motif/residual families.  This
    prevents the selector from overfitting hundreds of nearly-duplicated tuning
    variants while still giving it access to candidates that V20.7 showed were
    useful in oracle analyses.
    """
    if metrics is None or metrics.empty or "method" not in metrics.columns:
        return []
    available = list(pd.Series(metrics["method"].dropna().astype(str).unique()).values)
    exact_preferred = [
        "linear",
        "heading_hermite",
        "rtg_bridge",
        "brownian_bridge",
        "pretrained_motif_robust_global",
        "pretrained_motif_guarded",
        "pretrained_motif_adaptive_selector_v2",
        "pretrained_motif_two_sided_flow",
        "pretrained_motif_weighted_cost_rank_K20_b0.35",
        "pretrained_motif_weighted_cost_rank_K20_b0.5",
        "pretrained_motif_weighted_probability_K20_b0.25",
        "pretrained_motif_weighted_probability_K20_b0.35",
        "pretrained_motif_top1",
    ]
    out = [m for m in exact_preferred if m in available]
    # Keep only compact residual families that were meaningful in V20.7.
    residual_patterns = [
        r"^pretrained_motif_residual_linear_weighted_cost_rank_K20_b0\.?35_lamconf$",
        r"^pretrained_motif_residual_linear_weighted_cost_rank_K20_b0\.?5_lamconf$",
        r"^pretrained_motif_residual_flow_weighted_cost_rank_K20_b0\.?35_lamconf$",
        r"^pretrained_motif_residual_flow_weighted_cost_rank_K20_b0\.?5_lamconf$",
        r"^pretrained_motif_residual_flow_top1_phase_g0\.?9_lamconf$",
        r"^pretrained_motif_residual_flow_top1_phase_g1\.?1_lamconf$",
        r"^pretrained_motif_residual_linear_top1_phase_g0\.?9_lamconf$",
        r"^pretrained_motif_residual_linear_top1_phase_g1\.?1_lamconf$",
    ]
    for m in available:
        if m in out:
            continue
        if any(re.match(pat, m) for pat in residual_patterns):
            out.append(m)
    # If the compact list is too small, keep the strongest available proposed
    # methods but still avoid coherence-guard tuning variants.
    if len(out) < 5:
        for m in available:
            if m not in out and (m in {"linear", "heading_hermite", "rtg_bridge", "brownian_bridge"} or (_method_is_proposed_candidate(m) and "guarded_q" not in m)):
                out.append(m)
            if len(out) >= 12:
                break
    return out


def _oracle_distilled_feature_frame(rows: pd.DataFrame, train_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Build deployable features for the oracle-distilled selector.

    Excludes all evaluation/ground-truth error columns.  Candidate path length,
    confidence, source relation and cost diagnostics are allowed because they are
    known at reconstruction time.
    """
    if rows is None or rows.empty:
        return pd.DataFrame(), [] if train_columns is None else train_columns
    df = rows.copy()
    # Outcome / truth-derived columns that must never be features.
    blocked_exact = {
        "ADE", "RMSE", "Frechet", "DTW", "spatial_RMSE", "spatial_rmse_m", "time_indexed_rmse_m",
        "linear_ADE", "linear_RMSE", "linear_Frechet", "linear_DTW", "delta_ADE_vs_linear",
        "better_than_linear", "ade_ratio_to_linear", "rmse_ratio_to_linear", "frechet_ratio_to_linear", "dtw_ratio_to_linear",
        "ade_gain_pct_vs_linear", "rmse_gain_pct_vs_linear", "frechet_gain_pct_vs_linear", "dtw_gain_pct_vs_linear",
        "path_length_log_error", "path_length_ratio_error", "directness_error",
        "truth_directness", "truth_path_length_m", "truth_path_ratio", "truth_x", "truth_y",
    }
    blocked_tokens = [
        "abs_error", "error_", "_error", "gain_pct", "ratio_to_linear", "oracle", "truth_",
        "delta_", "better_than", "RMSE", "Frechet", "DTW", "ADE",
    ]
    safe_numeric = []
    for c in df.columns:
        if c in blocked_exact:
            continue
        if any(tok in str(c) for tok in blocked_tokens):
            continue
        if c in {"task_uid", "split", "start_time", "end_time"}:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            safe_numeric.append(c)
    cat_cols = [
        c for c in [
            "method", "dataset", "taxon", "setting_name", "habitat_id", "study_system",
            "species_id", "genus_group", "transfer_unit", "sex", "age_class",
            "source_method", "residual_source", "residual_baseline",
            "top_source_transfer_relation", "dominant_source_transfer_relation",
            "top_source_dataset", "top_source_taxon", "top_source_habitat_id",
        ] if c in df.columns
    ]
    parts = []
    if safe_numeric:
        num = df[safe_numeric].apply(pd.to_numeric, errors="coerce")
        num = num.replace([np.inf, -np.inf], np.nan)
        num = num.fillna(num.median(numeric_only=True)).fillna(0.0)
        parts.append(num.astype(float))
    for c in cat_cols:
        vals = df[c].fillna("missing").astype(str)
        d = pd.get_dummies(vals, prefix=c, dummy_na=False, dtype=float)
        parts.append(d)
    if parts:
        X = pd.concat(parts, axis=1)
    else:
        X = pd.DataFrame(index=df.index)
        X["bias"] = 1.0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if train_columns is None:
        cols = list(X.columns)
        return X, cols
    # Align prediction matrix to training feature columns.
    for c in train_columns:
        if c not in X.columns:
            X[c] = 0.0
    X = X[train_columns]
    return X, train_columns


def _fit_oracle_distilled_regressor(X: pd.DataFrame, y: np.ndarray):
    """Fit a small fast regressor; fall back gracefully if sklearn is limited."""
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=np.nanmedian(y[np.isfinite(y)]) if np.isfinite(y).any() else 0.0, posinf=2.0, neginf=-2.0)
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.05, random_state=42)
        model.fit(X, y)
        return model, "HistGradientBoostingRegressor"
    except Exception:
        try:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=120, max_depth=5, min_samples_leaf=3, random_state=42, n_jobs=1)
            model.fit(X, y)
            return model, "RandomForestRegressor"
        except Exception:
            # Last-resort method-level median predictor.
            return None, "method_median_fallback"


def _select_rows_by_predicted_score(source: pd.DataFrame, score_col: str, selector_name: str, linear_margin: float = 0.0) -> pd.DataFrame:
    if source is None or source.empty or score_col not in source.columns:
        return pd.DataFrame()
    rows = []
    for uid, g in source.groupby("task_uid", sort=False):
        gg = g.copy()
        gg[score_col] = pd.to_numeric(gg[score_col], errors="coerce")
        gg = gg[gg[score_col].notna()].copy()
        if gg.empty:
            continue
        # Best baseline fallback known at prediction time.
        baseline = gg[gg["method"].astype(str).isin(["linear", "heading_hermite"])].copy()
        if baseline.empty:
            baseline = gg[gg["method"].astype(str).isin(["linear", "heading_hermite", "rtg_bridge", "brownian_bridge"])].copy()
        best = gg.sort_values([score_col, "method"], ascending=[True, True], kind="mergesort").iloc[0].copy()
        if not baseline.empty and str(best.get("method")) not in set(baseline["method"].astype(str)):
            b = baseline.sort_values([score_col, "method"], ascending=[True, True], kind="mergesort").iloc[0].copy()
            # If the model only predicts a tiny gain over the simple baseline, use
            # the baseline to avoid overfitting motif/residual variants.
            if float(best[score_col]) > float(b[score_col]) - float(linear_margin):
                best = b
        source_method = str(best.get("method", "unknown"))
        best["method"] = selector_name
        best["source_method"] = source_method
        best["oracle_distilled_predicted_log_ratio"] = float(best[score_col])
        best["oracle_distilled_linear_margin"] = float(linear_margin)
        best["selector_version"] = "v20.8_oracle_distilled"
        rows.append(best.to_dict())
    return pd.DataFrame(rows)


def _score_selector_on_validation(metrics: pd.DataFrame, selector_rows: pd.DataFrame, selector_name: str, validation_split: str) -> dict:
    if selector_rows is None or selector_rows.empty:
        return {"med_ratio": np.inf, "mean_ratio": np.inf, "btl": -np.inf, "med_ade": np.inf}
    val_linear = metrics[(metrics["split"].astype(str).eq(validation_split)) & (metrics["method"].astype(str).eq("linear"))].copy()
    val_sel = selector_rows[selector_rows["split"].astype(str).eq(validation_split)].copy() if "split" in selector_rows.columns else selector_rows.copy()
    paired = add_linear_baseline_metrics(pd.concat([val_linear, val_sel], ignore_index=True, sort=False))
    sel = paired[paired["method"].astype(str).eq(selector_name)].copy()
    if sel.empty:
        return {"med_ratio": np.inf, "mean_ratio": np.inf, "btl": -np.inf, "med_ade": np.inf}
    ratio = pd.to_numeric(sel.get("ade_ratio_to_linear", pd.Series(dtype=float)), errors="coerce")
    ade = pd.to_numeric(sel.get("ADE", pd.Series(dtype=float)), errors="coerce")
    lin = pd.to_numeric(sel.get("linear_ADE", pd.Series(dtype=float)), errors="coerce")
    return {
        "med_ratio": float(ratio.median()) if ratio.notna().any() else np.inf,
        "mean_ratio": float(ratio.mean()) if ratio.notna().any() else np.inf,
        "btl": float((ade < lin).mean()) if lin.notna().any() else -np.inf,
        "med_ade": float(ade.median()) if ade.notna().any() else np.inf,
    }


def make_oracle_distilled_selection(
    metrics: pd.DataFrame,
    validation_split: str = "validation",
    selector_name: str = ORACLE_DISTILLED_SELECTOR_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train a validation-only oracle-distilled task selector.

    The selector predicts candidate log(ADE/linear) from deployable features and
    selects the method with the lowest predicted value per task.  The training
    target uses validation ADE only.  Test ADE remains unseen when selecting test
    rows, because the fitted model uses only feature columns.
    """
    if metrics is None or metrics.empty or "split" not in metrics.columns or "task_uid" not in metrics.columns or "method" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    metrics = add_linear_baseline_metrics(metrics.copy())
    pool_methods = _oracle_distilled_candidate_methods(metrics)
    if not pool_methods:
        return pd.DataFrame(), pd.DataFrame()
    pool = metrics[metrics["method"].astype(str).isin(pool_methods)].copy()
    if pool.empty:
        return pd.DataFrame(), pd.DataFrame()
    val = pool[pool["split"].astype(str).eq(validation_split)].copy()
    if val.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Target = log ADE ratio to linear. Lower is better.
    if "ade_ratio_to_linear" not in val.columns:
        val = add_linear_baseline_metrics(pd.concat([metrics[metrics["method"].eq("linear")], val], ignore_index=True, sort=False))
    y = pd.to_numeric(val.get("ade_ratio_to_linear", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    y = np.log(np.clip(y, 1e-3, 10.0))
    ok = np.isfinite(y)
    val_train = val.loc[ok].copy()
    y = y[ok]
    if len(val_train) < 10 or len(pd.Series(val_train["method"]).unique()) < 2:
        return pd.DataFrame(), pd.DataFrame()
    X_train, cols = _oracle_distilled_feature_frame(val_train, train_columns=None)
    model, model_name = _fit_oracle_distilled_regressor(X_train, y)
    source = pool.copy()
    X_all, _ = _oracle_distilled_feature_frame(source, train_columns=cols)
    if model is None:
        # Fallback prediction = validation median by method.
        med = pd.DataFrame({"method": val_train["method"].astype(str), "y": y}).groupby("method")["y"].median().to_dict()
        global_med = float(np.nanmedian(y)) if np.isfinite(y).any() else 0.0
        source["oracle_distilled_pred_raw"] = source["method"].astype(str).map(med).fillna(global_med).astype(float)
    else:
        source["oracle_distilled_pred_raw"] = np.asarray(model.predict(X_all), dtype=float)
    # Tune a small conservative fallback margin on validation. Units are log ADE ratio.
    margin_grid = [0.0, 0.005, 0.010, 0.020, 0.035, 0.050, 0.075]
    scored = []
    for margin in margin_grid:
        rows = _select_rows_by_predicted_score(source, "oracle_distilled_pred_raw", selector_name, linear_margin=margin)
        score = _score_selector_on_validation(metrics, rows, selector_name, validation_split)
        # Tie-break: prefer not collapsing entirely to complex methods unless validation improves.
        use_source = rows[rows.get("split", pd.Series(index=rows.index, dtype=str)).astype(str).eq(validation_split)] if "split" in rows.columns else rows
        non_baseline_rate = float((~use_source.get("source_method", pd.Series(dtype=str)).astype(str).isin(["linear", "heading_hermite"])).mean()) if not use_source.empty else np.nan
        scored.append({"margin": float(margin), **score, "non_baseline_rate": non_baseline_rate})
    score_df = pd.DataFrame(scored)
    score_df = score_df.sort_values(["med_ratio", "mean_ratio", "btl", "med_ade", "non_baseline_rate"], ascending=[True, True, False, True, True], kind="mergesort")
    best = score_df.iloc[0]
    out = _select_rows_by_predicted_score(source, "oracle_distilled_pred_raw", selector_name, linear_margin=float(best["margin"]))
    if out.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Add the paired metrics back to selector rows for immediate summaries.
    out = add_linear_baseline_metrics(pd.concat([metrics[metrics["method"].astype(str).eq("linear")], out], ignore_index=True, sort=False))
    out = out[out["method"].astype(str).eq(selector_name)].copy()
    choice = pd.DataFrame([{
        "paper_method": selector_name,
        "selected_method": "task_specific_oracle_distilled",
        "candidate_pool_n_methods": int(len(pool_methods)),
        "candidate_pool_methods": ";".join(pool_methods),
        "validation_training_rows": int(len(val_train)),
        "validation_training_tasks": int(val_train["task_uid"].nunique()),
        "selector_model": model_name,
        "selector_margin": float(best["margin"]),
        "validation_ADE_ratio_median": float(best["med_ratio"]),
        "validation_ADE_ratio_mean": float(best["mean_ratio"]),
        "validation_better_than_linear_rate": float(best["btl"]),
        "validation_ADE_median": float(best["med_ade"]),
        "validation_non_baseline_rate": float(best["non_baseline_rate"]),
        "selection_reason": "v20_8_oracle_distilled_validation_ranker",
    }])
    return out, choice

# V20.8 final speed override: use a small ridge regressor instead of tree ensembles.
def _fit_oracle_distilled_regressor(X: pd.DataFrame, y: np.ndarray):
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=np.nanmedian(y[np.isfinite(y)]) if np.isfinite(y).any() else 0.0, posinf=2.0, neginf=-2.0)
    try:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X, y)
        return model, "Ridge"
    except Exception:
        return None, "method_median_fallback"


# -----------------------------------------------------------------------------
# V20.9 conservative oracle-gap mitigation selector
# -----------------------------------------------------------------------------
# V20.8 reduced large outliers but switched away from the guarded default too
# often.  V20.9 reframes selection as a safer binary decision:
#   keep the default full-coverage method unless a candidate is confidently
#   predicted to beat that default on a task.
# This uses validation labels only for fitting/tuning and deployable features at
# prediction time. It is intentionally conservative and fast.

CONSERVATIVE_ORACLE_SELECTOR_NAME = "pretrained_motif_conservative_oracle_switcher"


def _v209_default_method(metrics: pd.DataFrame) -> str:
    available = set(metrics.get("method", pd.Series(dtype=str)).dropna().astype(str))
    for m in [
        "pretrained_motif_guarded",
        "pretrained_motif_robust_global",
        "pretrained_motif_adaptive_selector_v2",
        "pretrained_motif_oracle_distilled_selector",
        "heading_hermite",
        "linear",
    ]:
        if m in available:
            return m
    if available:
        return sorted(available)[0]
    return "linear"


def _v209_candidate_pool_methods(metrics: pd.DataFrame, default_method: str) -> list[str]:
    """Small candidate set for conservative switching.

    Keeps the methods that were informative in V20.7/V20.8 while avoiding the
    hundreds of near-duplicate tuning variants.  The selector can only switch to
    one of these methods; otherwise it keeps the default.
    """
    available = list(pd.Series(metrics.get("method", pd.Series(dtype=str)).dropna().astype(str).unique()).values)
    exact = [
        "linear",
        "heading_hermite",
        "rtg_bridge",
        "brownian_bridge",
        default_method,
        "pretrained_motif_guarded",
        "pretrained_motif_robust_global",
        "pretrained_motif_oracle_distilled_selector",
        "pretrained_motif_adaptive_selector_v2",
        "pretrained_motif_two_sided_flow",
        "pretrained_motif_weighted_cost_rank_K20_b0.35",
        "pretrained_motif_weighted_cost_rank_K20_b0.5",
        "pretrained_motif_weighted_probability_K20_b0.25",
        "pretrained_motif_top1_phase_g0.9",
        "pretrained_motif_top1_phase_g1.1",
        "pretrained_motif_residual_linear_weighted_cost_rank_K20_b0.35_lamconf",
        "pretrained_motif_residual_linear_weighted_cost_rank_K20_b0.5_lamconf",
        "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.35_lamconf",
        "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.5_lamconf",
        "pretrained_motif_residual_flow_top1_phase_g0.9_lamconf",
        "pretrained_motif_residual_flow_top1_phase_g1.1_lamconf",
    ]
    out = []
    for m in exact:
        if m in available and m not in out:
            out.append(m)
    # Add at most a few phase/residual variants if exact names differ slightly.
    regex_keep = [
        r"^pretrained_motif_top1_phase_g(0\.9|1\.1)$",
        r"^pretrained_motif_residual_(linear|flow)_weighted_cost_rank_K20_b(0\.35|0\.5)_lamconf$",
        r"^pretrained_motif_residual_flow_top1_phase_g(0\.9|1\.1)_lamconf$",
    ]
    for m in available:
        if m in out:
            continue
        if any(re.match(pat, m) for pat in regex_keep):
            out.append(m)
    return out


def _v209_attach_default_metrics(pool: pd.DataFrame, default_method: str) -> pd.DataFrame:
    default = pool[pool["method"].astype(str).eq(default_method)].copy()
    if default.empty and default_method != "linear":
        default = pool[pool["method"].astype(str).eq("linear")].copy()
        default_method = "linear"
    if default.empty:
        return pd.DataFrame()
    keep = ["task_uid", "ADE", "RMSE", "ade_ratio_to_linear", "rmse_ratio_to_linear", "better_than_linear"]
    keep = [c for c in keep if c in default.columns]
    d = default[keep].drop_duplicates("task_uid").copy()
    rename = {c: f"default_{c}" for c in keep if c != "task_uid"}
    d = d.rename(columns=rename)
    out = pool.merge(d, on="task_uid", how="left")
    out["v209_default_method"] = default_method
    return out


def _v209_train_pairwise_switcher(train_rows: pd.DataFrame, min_gain_ratio: float = 0.005):
    """Fit a lightweight classifier: candidate clearly beats default?"""
    if train_rows.empty or "default_ADE" not in train_rows.columns:
        return None, None, [], "unavailable"
    y_raw = pd.to_numeric(train_rows["ADE"], errors="coerce") < (pd.to_numeric(train_rows["default_ADE"], errors="coerce") * (1.0 - float(min_gain_ratio)))
    ok = y_raw.notna()
    yy = y_raw.loc[ok].astype(int).to_numpy()
    tr = train_rows.loc[ok].copy()
    if len(tr) < 20 or len(np.unique(yy)) < 2:
        return None, None, [], "method_gain_fallback"
    X, cols = _oracle_distilled_feature_frame(tr, train_columns=None)
    try:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=500, class_weight="balanced", C=0.5, solver="liblinear", random_state=42)
        model.fit(X, yy)
        return model, None, cols, "LogisticRegression_pairwise_switch"
    except Exception:
        # Fallback: validation win rate by method.
        rates = pd.DataFrame({"method": tr["method"].astype(str), "win": yy}).groupby("method")["win"].mean().to_dict()
        return None, rates, [], "method_gain_rate_fallback"


def _v209_predict_switch_probability(rows: pd.DataFrame, model, method_rates: dict | None, cols: list[str]) -> np.ndarray:
    if rows.empty:
        return np.array([], dtype=float)
    if model is not None and cols:
        X, _ = _oracle_distilled_feature_frame(rows, train_columns=cols)
        try:
            proba = model.predict_proba(X)[:, 1]
            return np.asarray(proba, dtype=float)
        except Exception:
            pass
    if method_rates:
        return rows["method"].astype(str).map(method_rates).fillna(0.0).astype(float).to_numpy()
    return np.zeros(len(rows), dtype=float)


def _v209_select_rows(pool: pd.DataFrame, default_method: str, selector_name: str, proba_col: str, threshold: float, min_pred_margin: float = 0.0) -> pd.DataFrame:
    """Keep default unless candidate switch probability clears threshold."""
    if pool.empty:
        return pd.DataFrame()
    default = pool[pool["method"].astype(str).eq(default_method)].copy()
    if default.empty and default_method != "linear":
        default = pool[pool["method"].astype(str).eq("linear")].copy()
    default_map = default.set_index("task_uid", drop=False) if not default.empty else None
    rows = []
    for uid, g in pool.groupby("task_uid", sort=False):
        gg = g.copy()
        # Do not switch to the default itself; it is the fallback row.
        cand = gg[~gg["method"].astype(str).eq(default_method)].copy()
        cand = cand[pd.to_numeric(cand.get(proba_col, pd.Series(index=cand.index, dtype=float)), errors="coerce").ge(float(threshold))]
        if not cand.empty:
            # Prefer high probability; tie-break by lower predicted oracle ratio when available and simpler methods.
            sort_cols = [proba_col]
            ascending = [False]
            if "oracle_distilled_pred_raw" in cand.columns:
                sort_cols.append("oracle_distilled_pred_raw")
                ascending.append(True)
            if "selector_confidence" in cand.columns:
                sort_cols.append("selector_confidence")
                ascending.append(False)
            chosen = cand.sort_values(sort_cols, ascending=ascending, kind="mergesort").iloc[0].copy()
            chosen["selector_switched_from_default"] = 1
            chosen["selector_source_method"] = str(chosen.get("method", ""))
        else:
            if default_map is not None and uid in default_map.index:
                chosen = default_map.loc[uid].copy()
                if isinstance(chosen, pd.DataFrame):
                    chosen = chosen.iloc[0].copy()
            else:
                chosen = gg.iloc[0].copy()
            chosen["selector_switched_from_default"] = 0
            chosen["selector_source_method"] = str(chosen.get("method", ""))
        chosen["method"] = selector_name
        chosen["selector_default_method"] = default_method
        chosen["selector_probability_threshold"] = float(threshold)
        rows.append(chosen.to_dict())
    return pd.DataFrame(rows)


def _v209_score_selection(all_metrics: pd.DataFrame, rows: pd.DataFrame, selector_name: str, split: str = "validation") -> dict:
    if rows is None or rows.empty:
        return {"med_ratio": np.inf, "mean_ratio": np.inf, "btl": -np.inf, "mean_ade": np.inf, "switch_rate": np.inf}
    base = all_metrics[all_metrics["method"].astype(str).eq("linear")].copy()
    tmp = add_linear_baseline_metrics(pd.concat([base, rows], ignore_index=True, sort=False))
    sub = tmp[tmp["method"].astype(str).eq(selector_name)].copy()
    if "split" in sub.columns:
        sub = sub[sub["split"].astype(str).eq(split)]
    if sub.empty:
        return {"med_ratio": np.inf, "mean_ratio": np.inf, "btl": -np.inf, "mean_ade": np.inf, "switch_rate": np.inf}
    ratio = pd.to_numeric(sub.get("ade_ratio_to_linear", pd.Series(index=sub.index, dtype=float)), errors="coerce")
    ade = pd.to_numeric(sub.get("ADE", pd.Series(index=sub.index, dtype=float)), errors="coerce")
    btl = pd.to_numeric(sub.get("better_than_linear", pd.Series(index=sub.index, dtype=float)), errors="coerce")
    sw = pd.to_numeric(sub.get("selector_switched_from_default", pd.Series(index=sub.index, dtype=float)), errors="coerce")
    return {
        "med_ratio": float(ratio.median()) if ratio.notna().any() else np.inf,
        "mean_ratio": float(ratio.mean()) if ratio.notna().any() else np.inf,
        "btl": float(btl.mean()) if btl.notna().any() else -np.inf,
        "mean_ade": float(ade.mean()) if ade.notna().any() else np.inf,
        "switch_rate": float(sw.mean()) if sw.notna().any() else np.inf,
    }


def make_conservative_oracle_gap_selection(
    metrics: pd.DataFrame,
    validation_split: str = "validation",
    selector_name: str = CONSERVATIVE_ORACLE_SELECTOR_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation-trained conservative oracle-gap switcher.

    Default-preserving task selector:
      1. choose default = guarded if available, else robust_global/linear;
      2. train a candidate-vs-default switch classifier on validation rows;
      3. tune the probability threshold on validation to optimize median ADE ratio
         and win rate while discouraging excessive switching;
      4. apply the threshold to all splits.
    """
    if metrics is None or metrics.empty or "method" not in metrics.columns or "task_uid" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    mm = add_linear_baseline_metrics(metrics.copy())
    default_method = _v209_default_method(mm)
    pool_methods = _v209_candidate_pool_methods(mm, default_method)
    if default_method not in pool_methods:
        pool_methods.insert(0, default_method)
    pool = mm[mm["method"].astype(str).isin(pool_methods)].copy()
    if pool.empty or default_method not in set(pool["method"].astype(str)):
        return pd.DataFrame(), pd.DataFrame()
    pool = _v209_attach_default_metrics(pool, default_method)
    if pool.empty:
        return pd.DataFrame(), pd.DataFrame()
    val = pool[pool.get("split", pd.Series(index=pool.index, dtype=str)).astype(str).eq(validation_split)].copy()
    if val.empty:
        return pd.DataFrame(), pd.DataFrame()
    model, method_rates, cols, model_name = _v209_train_pairwise_switcher(val, min_gain_ratio=0.005)
    pool["v209_switch_probability"] = _v209_predict_switch_probability(pool, model, method_rates, cols)
    # Tune conservative threshold. Higher thresholds reduce over-switching.
    threshold_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    scored = []
    for thr in threshold_grid:
        rows = _v209_select_rows(pool, default_method, selector_name, "v209_switch_probability", threshold=thr)
        score = _v209_score_selection(mm, rows, selector_name, split=validation_split)
        scored.append({"threshold": float(thr), **score})
    score_df = pd.DataFrame(scored)
    # Validation objective: median ratio first, then win rate, then mean ratio,
    # then lower mean ADE and lower switch rate for robustness.
    score_df = score_df.sort_values(["med_ratio", "btl", "mean_ratio", "mean_ade", "switch_rate"], ascending=[True, False, True, True, True], kind="mergesort")
    best = score_df.iloc[0]
    rows = _v209_select_rows(pool, default_method, selector_name, "v209_switch_probability", threshold=float(best["threshold"]))
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    out = add_linear_baseline_metrics(pd.concat([mm[mm["method"].astype(str).eq("linear")], rows], ignore_index=True, sort=False))
    out = out[out["method"].astype(str).eq(selector_name)].copy()
    choice = pd.DataFrame([{
        "paper_method": selector_name,
        "selected_method": "task_specific_conservative_oracle_switch",
        "default_method": default_method,
        "candidate_pool_n_methods": int(len(pool_methods)),
        "candidate_pool_methods": ";".join(pool_methods),
        "validation_training_rows": int(len(val)),
        "validation_training_tasks": int(val["task_uid"].nunique()),
        "selector_model": model_name,
        "selector_threshold": float(best["threshold"]),
        "validation_ADE_ratio_median": float(best["med_ratio"]),
        "validation_ADE_ratio_mean": float(best["mean_ratio"]),
        "validation_better_than_linear_rate": float(best["btl"]),
        "validation_ADE_mean": float(best["mean_ade"]),
        "validation_switch_rate": float(best["switch_rate"]),
        "selection_reason": "v20_9_conservative_oracle_gap_switcher",
    }])
    return out, choice


def compact_v209_metrics_for_reporting(metrics: pd.DataFrame) -> pd.DataFrame:
    """Optional helper to keep only paper-facing compact method rows.

    Not used by default inside the benchmark because we still want diagnostics;
    useful in notebooks if summaries become too crowded.
    """
    if metrics is None or metrics.empty or "method" not in metrics.columns:
        return metrics
    keep_exact = {
        "linear", "heading_hermite", "rtg_bridge", "brownian_bridge",
        "pretrained_motif_guarded", "pretrained_motif_robust_global",
        "pretrained_motif_oracle_distilled_selector",
        CONSERVATIVE_ORACLE_SELECTOR_NAME,
    }
    method = metrics["method"].astype(str)
    keep = method.isin(keep_exact)
    # Keep a few residual diagnostics if present.
    keep |= method.str.contains("residual_flow_weighted_cost_rank_K20_b0.35_lamconf", regex=False)
    keep |= method.str.contains("residual_linear_weighted_cost_rank_K20_b0.35_lamconf", regex=False)
    keep |= method.str.contains("v26", regex=False)
    return metrics[keep].copy()



# -----------------------------------------------------------------------------
# V21d/e/f shared one-day selector helpers
# -----------------------------------------------------------------------------
# These helpers deliberately operate on already-evaluated candidate rows. They
# do not inspect test truth during prediction. Validation is used only to choose
# setting-level methods or tune a conservative switch margin.

def _v21_safe_num_series(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if frame is None or frame.empty or col not in frame.columns:
        return pd.Series(default, index=frame.index if frame is not None else None, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _v21_group_cols(metrics: pd.DataFrame) -> list[str]:
    cols = [c for c in ["dataset", "taxon", "setting_name"] if c in metrics.columns]
    return cols or ["setting_name"] if "setting_name" in metrics.columns else []


def _v21_default_rows(metrics: pd.DataFrame, preferred: str = "pretrained_motif_robust_global") -> tuple[str, pd.DataFrame]:
    vals = set(metrics.get("method", pd.Series(dtype=str)).dropna().astype(str).unique())
    order = [preferred, "pretrained_motif_guarded", "pretrained_motif_conservative_oracle_switcher", "linear"]
    for m in order:
        if m in vals:
            return m, metrics[metrics["method"].astype(str).eq(m)].copy()
    m = str(metrics["method"].dropna().astype(str).iloc[0]) if "method" in metrics.columns and not metrics.empty else "linear"
    return m, metrics[metrics["method"].astype(str).eq(m)].copy()


def _v21_candidate_method_set(metrics: pd.DataFrame, residual_only: bool = False) -> set[str]:
    vals = set(metrics.get("method", pd.Series(dtype=str)).dropna().astype(str).unique())
    out = set()
    bad_substrings = [
        "endpoint_fallback", "oracle_distilled", "conservative_oracle", "legacy_setting_selector",
        "gb_ranker", "gb_residual", "v21", "paper",
    ]
    for m in vals:
        s = str(m)
        if s in {"linear", "heading_hermite", "brownian_bridge", "rtg_bridge"}:
            continue
        if any(b in s for b in bad_substrings):
            continue
        if not s.startswith("pretrained_motif_"):
            continue
        if residual_only and ("residual_" not in s and "two_sided_flow" not in s and "robust_global" not in s and "guarded" not in s):
            continue
        out.add(s)
    # Always allow the stable global/guarded choices if present.
    for m in ["pretrained_motif_robust_global", "pretrained_motif_guarded"]:
        if m in vals:
            out.add(m)
    return out


def _v21_score_selection_rows(mm: pd.DataFrame, rows: pd.DataFrame, selector_name: str, split: str = "validation") -> dict:
    if rows is None or rows.empty:
        return {"med_ratio": np.inf, "mean_ratio": np.inf, "btl": -np.inf, "mean_ade": np.inf, "n": 0}
    tmp = add_linear_baseline_metrics(pd.concat([mm[mm["method"].astype(str).eq("linear")], rows], ignore_index=True, sort=False))
    sel = tmp[tmp["method"].astype(str).eq(selector_name)].copy()
    if "split" in sel.columns:
        sel = sel[sel["split"].astype(str).eq(split)].copy()
    if sel.empty:
        return {"med_ratio": np.inf, "mean_ratio": np.inf, "btl": -np.inf, "mean_ade": np.inf, "n": 0}
    ratio = pd.to_numeric(sel.get("ade_ratio_to_linear", pd.Series(index=sel.index, dtype=float)), errors="coerce")
    ade = pd.to_numeric(sel.get("ADE", pd.Series(index=sel.index, dtype=float)), errors="coerce")
    lin = pd.to_numeric(sel.get("linear_ADE", pd.Series(index=sel.index, dtype=float)), errors="coerce")
    return {
        "med_ratio": float(ratio.median()) if ratio.notna().any() else np.inf,
        "mean_ratio": float(ratio.mean()) if ratio.notna().any() else np.inf,
        "btl": float((ade < lin).mean()) if lin.notna().any() else -np.inf,
        "mean_ade": float(ade.mean()) if ade.notna().any() else np.inf,
        "n": int(sel["task_uid"].nunique()) if "task_uid" in sel.columns else int(len(sel)),
    }


def _v21_copy_method_rows_for_tasks(metrics: pd.DataFrame, method_by_group: dict, selector_name: str, group_cols: list[str]) -> pd.DataFrame:
    if metrics is None or metrics.empty or not group_cols:
        return pd.DataFrame()
    rows = []
    for _, g in metrics.groupby(group_cols, dropna=False, sort=False):
        key = tuple(str(g[c].iloc[0]) for c in group_cols)
        selected = method_by_group.get(key)
        if selected is None:
            continue
        src = g[g["method"].astype(str).eq(str(selected))].copy()
        if src.empty:
            continue
        for _, r in src.drop_duplicates("task_uid").iterrows():
            rr = r.copy()
            rr["method"] = selector_name
            rr["v21_selected_source_method"] = str(selected)
            rr["source_method"] = str(selected)
            rr["selector_version"] = selector_name
            rows.append(rr.to_dict())
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# V21f: residual-focused gradient-boosted rescue selector
# -----------------------------------------------------------------------------
V21F_GB_RESIDUAL_SELECTOR_NAME = "pretrained_motif_gb_residual_rescue_selector"


def make_v21f_gb_residual_rescue_selector(
    metrics: pd.DataFrame,
    validation_split: str = "validation",
    selector_name: str = V21F_GB_RESIDUAL_SELECTOR_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residual-focused ML rescue selector.

    This is the one-day lightweight version of learned residual correction: it
    learns, on validation, when an already generated residual-flow/linear
    residual reconstruction should replace robust_global. It is intentionally
    conservative and residual-only so it does not chase unrelated candidate
    families.
    """
    if metrics is None or metrics.empty or "method" not in metrics.columns or "task_uid" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    mm = add_linear_baseline_metrics(metrics.copy())
    default_method, default_rows = _v21_default_rows(mm, preferred="pretrained_motif_robust_global")
    candidates = _v21_candidate_method_set(mm, residual_only=True)
    # Ensure a baseline default and guarded are in the pool.
    candidates.add(default_method)
    if "pretrained_motif_guarded" in set(mm["method"].astype(str)):
        candidates.add("pretrained_motif_guarded")
    pool = mm[mm["method"].astype(str).isin(candidates)].copy()
    if pool.empty or default_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    train = pool[pool.get("split", pd.Series(index=pool.index, dtype=str)).astype(str).eq(validation_split)].copy()
    train = train[pd.to_numeric(train.get("ade_ratio_to_linear", np.nan), errors="coerce").notna()].copy()
    if train["task_uid"].nunique() < 8 or len(train) < 20:
        return pd.DataFrame(), pd.DataFrame()
    # Reuse V21e feature builder if present; otherwise build minimal residual features.
    if "_v21e_feature_frame" in globals():
        X_all = _v21e_feature_frame(pool)
    else:
        X_all = pd.DataFrame(index=pool.index)
        s = pool["method"].astype(str)
        X_all["is_residual"] = s.str.contains("residual", regex=False).astype(float)
        X_all["is_flow"] = s.str.contains("flow", regex=False).astype(float)
        X_all["is_guarded"] = s.str.contains("guarded", regex=False).astype(float)
        X_all["is_robust"] = s.str.contains("robust_global", regex=False).astype(float)
        for c in ["residual_lambda", "selector_confidence", "adaptive_confidence", "source_dispersion_mean_m", "source_score_margin", "path_topK", "path_weight_beta"]:
            if c in pool.columns:
                X_all[c] = pd.to_numeric(pool[c], errors="coerce")
        X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    y = pd.to_numeric(train["ade_ratio_to_linear"], errors="coerce").to_numpy(dtype=float)
    pred = np.full(len(pool), np.nan, dtype=float)
    model_name = "residual_method_validation_median_fallback"
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=160, max_depth=6, min_samples_leaf=3, random_state=43, n_jobs=1)
        model.fit(X_all.loc[train.index], y)
        pred = model.predict(X_all)
        model_name = "ExtraTreesRegressor_residual_pool"
    except Exception:
        med_by_method = train.groupby("method")["ade_ratio_to_linear"].median().to_dict()
        global_med = float(np.nanmedian(y)) if len(y) else 1.0
        pred = pool["method"].astype(str).map(med_by_method).fillna(global_med).to_numpy(dtype=float)
    pool["v21f_predicted_ADE_ratio"] = pred

    def select_with_margin(margin: float, source: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for uid, g in source.groupby("task_uid", sort=False):
            default_g = g[g["method"].astype(str).eq(default_method)]
            default_row = default_g.iloc[0] if not default_g.empty else g.iloc[0]
            residual_g = g[g["method"].astype(str).str.contains("residual", regex=False)].copy()
            if residual_g.empty:
                best = default_row
                use = False
            else:
                residual_g["_pred"] = pd.to_numeric(residual_g["v21f_predicted_ADE_ratio"], errors="coerce")
                best = residual_g.sort_values(["_pred", "method"], ascending=[True, True], kind="mergesort").iloc[0]
                dp = float(default_row.get("v21f_predicted_ADE_ratio", np.nan))
                bp = float(best.get("v21f_predicted_ADE_ratio", np.nan))
                use = np.isfinite(dp) and np.isfinite(bp) and (bp < dp - float(margin))
            src = best if use else default_row
            rr = src.copy()
            rr["method"] = selector_name
            rr["v21_selected_source_method"] = str(src.get("method", ""))
            rr["source_method"] = str(src.get("method", ""))
            rr["v21f_predicted_ADE_ratio"] = float(src.get("v21f_predicted_ADE_ratio", np.nan))
            rr["v21f_switch_margin"] = float(margin)
            rr["selector_used_proposed"] = int(use)
            rr["selector_version"] = selector_name
            rows.append(rr.to_dict())
        return pd.DataFrame(rows)

    val_pool = pool[pool.get("split", pd.Series(index=pool.index, dtype=str)).astype(str).eq(validation_split)].copy()
    margins = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
    scored = []
    for mg in margins:
        rows = select_with_margin(mg, val_pool)
        score = _v21_score_selection_rows(mm, rows, selector_name, split=validation_split)
        switch_rate = float(pd.to_numeric(rows.get("selector_used_proposed", pd.Series(dtype=float)), errors="coerce").mean()) if not rows.empty else np.inf
        scored.append({"margin": float(mg), "switch_rate": switch_rate, **score})
    ss = pd.DataFrame(scored).sort_values(["med_ratio", "btl", "mean_ratio", "mean_ade", "switch_rate"], ascending=[True, False, True, True, True], kind="mergesort")
    best = ss.iloc[0]
    out = select_with_margin(float(best["margin"]), pool)
    out = add_linear_baseline_metrics(pd.concat([mm[mm["method"].astype(str).eq("linear")], out], ignore_index=True, sort=False))
    out = out[out["method"].astype(str).eq(selector_name)].copy()
    choice = pd.DataFrame([{
        "paper_method": selector_name,
        "default_method": default_method,
        "selected_method": "task_specific_residual_rescue",
        "selector_model": model_name,
        "selector_margin": float(best["margin"]),
        "candidate_pool_n_methods": int(len(candidates)),
        "validation_ADE_ratio_median": float(best["med_ratio"]),
        "validation_ADE_ratio_mean": float(best["mean_ratio"]),
        "validation_better_than_linear_rate": float(best["btl"]),
        "validation_ADE_mean": float(best["mean_ade"]),
        "validation_switch_rate": float(best["switch_rate"]),
        "selection_reason": "v21f_gradient_boosted_residual_rescue_selector",
    }])
    return out, choice



# =============================================================================
# V26 experimental selectors: borrow strongest V25h candidate-set behavior
# =============================================================================
# These selectors operate only on already evaluated rows generated by the normal
# candidate bank.  They never use held-out test ADE to select a method; test ADE is
# used only after the selected row has been copied for evaluation/reporting.

V26A_SELECTOR_NAME = "pretrained_motif_v26a_v25h_safety_gate"
V26B_SELECTOR_NAME = "pretrained_motif_v26b_regime_aware_selector"
V26C_PREFIX = "pretrained_motif_v26c"


def _v26_method_values(metrics: pd.DataFrame) -> set[str]:
    if metrics is None or metrics.empty or "method" not in metrics.columns:
        return set()
    return set(metrics["method"].dropna().astype(str).unique())


def _v26_is_selector_method(method: str) -> bool:
    m = str(method)
    selector_tokens = [
        "oracle_distilled", "conservative_oracle", "adaptive_selector", "gb_residual",
        "v21", "v26", "paper_method", "guarded_q", "legacy_setting_selector",
    ]
    return any(tok in m for tok in selector_tokens)


def _v26_proposed_nonselector_methods(metrics: pd.DataFrame, include_baseline_fallbacks: bool = False) -> list[str]:
    vals = sorted(_v26_method_values(metrics))
    out: list[str] = []
    for m in vals:
        if m in {"linear", "heading_hermite", "rtg_bridge", "brownian_bridge"}:
            if include_baseline_fallbacks and m in {"linear", "heading_hermite"}:
                out.append(m)
            continue
        if not m.startswith("pretrained_motif_"):
            continue
        if _v26_is_selector_method(m):
            continue
        if "endpoint_fallback" in m:
            continue
        out.append(m)
    return out


def _v26_v25h_weighted_candidate_methods(metrics: pd.DataFrame, include_safe: bool = True) -> list[str]:
    """Candidate method pool inspired by the old V25h weighted K20 result.

    The old V25h strongest median result came from K20 weighted representative
    paths, especially cost-rank weighted variants.  Here we keep those paths but
    let a safety gate or regime selector decide when they are trusted.
    """
    vals = _v26_method_values(metrics)
    preferred = [
        "pretrained_motif_weighted_cost_rank_K20_b0.25",
        "pretrained_motif_weighted_cost_rank_K20_b0.35",
        "pretrained_motif_weighted_cost_rank_K20_b0.5",
        "pretrained_motif_weighted_probability_K20_b0.25",
        "pretrained_motif_weighted_probability_K20_b0.35",
        "pretrained_motif_blend_direct_weighted_cost_rank_K20_b0.25_lam0.25",
        "pretrained_motif_blend_direct_weighted_cost_rank_K20_b0.35_lam0.25",
        "pretrained_motif_blend_direct_weighted_cost_rank_K20_b0.5_lam0.25",
        "pretrained_motif_blend_top1_weighted_cost_rank_K20_b0.35_lam0.25",
        "pretrained_motif_blend_top1_weighted_cost_rank_K20_b0.5_lam0.25",
        "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.35_lamconf",
        "pretrained_motif_residual_linear_weighted_cost_rank_K20_b0.35_lamconf",
    ]
    out = [m for m in preferred if m in vals]
    # Keep any exact naming variants not covered above.
    for m in sorted(vals):
        s = str(m)
        if s in out or not s.startswith("pretrained_motif_") or _v26_is_selector_method(s):
            continue
        if "endpoint_fallback" in s:
            continue
        if (("weighted_cost_rank_K20" in s or "weighted_probability_K20" in s or "blend_direct_weighted" in s)
                and "guarded_q" not in s):
            out.append(s)
    if include_safe:
        for m in [
            "pretrained_motif_robust_global",
            "pretrained_motif_guarded",
            "pretrained_motif_two_sided_flow",
            "linear",
            "heading_hermite",
        ]:
            if m in vals and m not in out:
                out.append(m)
    return out


def _v26_default_method(metrics: pd.DataFrame) -> str:
    vals = _v26_method_values(metrics)
    for m in ["pretrained_motif_robust_global", "pretrained_motif_guarded", "linear", "heading_hermite"]:
        if m in vals:
            return m
    if vals:
        return sorted(vals)[0]
    return "linear"


def _v26_attach_default(pool: pd.DataFrame, default_method: str) -> pd.DataFrame:
    if pool is None or pool.empty:
        return pd.DataFrame()
    default = pool[pool["method"].astype(str).eq(str(default_method))].copy()
    if default.empty and default_method != "linear":
        default = pool[pool["method"].astype(str).eq("linear")].copy()
    if default.empty:
        return pd.DataFrame()
    keep = [c for c in [
        "task_uid", "ADE", "RMSE", "Frechet", "DTW", "path_length_log_error",
        "ade_ratio_to_linear", "better_than_linear",
    ] if c in default.columns]
    d = default[keep].drop_duplicates("task_uid").copy()
    d = d.rename(columns={c: f"v26_default_{c}" for c in keep if c != "task_uid"})
    out = pool.merge(d, on="task_uid", how="left")
    out["v26_default_method"] = default_method
    return out


def _v26_safe_feature_frame(rows: pd.DataFrame, train_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Small fast deployable feature builder for V26a safety gate.

    Deliberately avoids high-cardinality object columns and any truth/evaluation
    columns so selector fitting stays fast after the expensive trajectory
    generation step.
    """
    df = rows.copy() if rows is not None else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(), [] if train_columns is None else train_columns
    numeric_candidates = [
        "path_topK", "path_weight_beta", "blend_lambda_weighted", "phase_gamma",
        "residual_lambda", "selector_confidence", "adaptive_confidence",
        "source_dispersion_mean_m", "source_score_margin", "n_source_candidates",
        "n_retrieved_source_candidates", "used_endpoint_fallback_only",
        "proposal_cost", "proposal_score", "cost_rank_score", "listwise_probability",
        "candidate_cost_context", "candidate_cost_direction", "candidate_cost_efficiency",
        "candidate_cost_source_shape", "candidate_cost_motif", "candidate_cost_step",
        "candidate_cost_turn", "candidate_cost_detour", "candidate_cost_directness",
        "candidate_cost_lateral", "candidate_cost_timegeo", "context_cost",
        "context_temporal_cost", "context_environment_cost", "context_demographic_cost",
        "context_n_environment_matches", "direction_cost", "efficiency_cost",
        "source_shape_cost", "source_path_ratio", "path_ratio", "expected_path_ratio",
        "target_path_ratio", "source_directness", "directness", "step_violation_fraction",
        "step_q90_over_capacity",
    ]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    parts = []
    if numeric_cols:
        x = df[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = x.median(numeric_only=True)
        parts.append(x.fillna(med).fillna(0.0).astype(float))
    cat_cols = [c for c in [
        "method", "dataset", "taxon", "setting_name", "species_id", "habitat_id",
        "best_train_transfer_relation", "source_method", "residual_source", "residual_baseline",
        "candidate_origin", "method_family_detail", "score_col",
    ] if c in df.columns]
    for c in cat_cols:
        vals = df[c].fillna("missing").astype(str)
        # Cap extremely rare categories to keep matrix compact and avoid slow fitting.
        vc = vals.value_counts(dropna=False)
        keep = set(vc[vc >= 2].index.astype(str))
        vals = vals.where(vals.isin(keep), other="__rare__")
        parts.append(pd.get_dummies(vals, prefix=c, dtype=float))
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame({"bias": np.ones(len(df))}, index=df.index)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if train_columns is None:
        return X, list(X.columns)
    for c in train_columns:
        if c not in X.columns:
            X[c] = 0.0
    return X[train_columns], train_columns


def _v26_fit_classifier(X: pd.DataFrame, y: np.ndarray):
    y = np.asarray(y, dtype=int)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None, "method_prior_fallback"
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(n_estimators=140, max_depth=7, min_samples_leaf=3, class_weight="balanced", random_state=2601, n_jobs=1)
        model.fit(X, y)
        return model, "ExtraTreesClassifier_safety_gate"
    except Exception:
        try:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=500, class_weight="balanced", C=0.5, solver="liblinear", random_state=2601)
            model.fit(X, y)
            return model, "LogisticRegression_safety_gate"
        except Exception:
            return None, "method_prior_fallback"


def _v26_predict_probability(rows: pd.DataFrame, model, cols: list[str], method_priors: dict[str, float]) -> np.ndarray:
    if rows is None or rows.empty:
        return np.array([], dtype=float)
    if model is not None and cols:
        try:
            X, _ = _v26_safe_feature_frame(rows, train_columns=cols)
            if hasattr(model, "predict_proba"):
                return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
        except Exception:
            pass
    return rows["method"].astype(str).map(method_priors).fillna(0.0).to_numpy(dtype=float)


def _v26_selection_score(mm: pd.DataFrame, rows: pd.DataFrame, selector_name: str, split: str = "validation") -> dict:
    if rows is None or rows.empty:
        return {"med_ratio": np.inf, "mean_ratio": np.inf, "q90_ratio": np.inf, "btl": -np.inf, "mean_ade": np.inf, "switch_rate": np.inf, "n": 0}
    base = mm[mm["method"].astype(str).eq("linear")].copy()
    tmp = add_linear_baseline_metrics(pd.concat([base, rows], ignore_index=True, sort=False))
    sel = tmp[tmp["method"].astype(str).eq(selector_name)].copy()
    if "split" in sel.columns:
        sel = sel[sel["split"].astype(str).eq(split)].copy()
    if sel.empty:
        return {"med_ratio": np.inf, "mean_ratio": np.inf, "q90_ratio": np.inf, "btl": -np.inf, "mean_ade": np.inf, "switch_rate": np.inf, "n": 0}
    ratio = pd.to_numeric(sel.get("ade_ratio_to_linear", pd.Series(index=sel.index, dtype=float)), errors="coerce")
    ade = pd.to_numeric(sel.get("ADE", pd.Series(index=sel.index, dtype=float)), errors="coerce")
    lin = pd.to_numeric(sel.get("linear_ADE", pd.Series(index=sel.index, dtype=float)), errors="coerce")
    sw = pd.to_numeric(sel.get("selector_used_risky", sel.get("selector_switched_from_default", pd.Series(index=sel.index, dtype=float))), errors="coerce")
    return {
        "med_ratio": float(ratio.median()) if ratio.notna().any() else np.inf,
        "mean_ratio": float(ratio.mean()) if ratio.notna().any() else np.inf,
        "q90_ratio": float(ratio.quantile(0.90)) if ratio.notna().any() else np.inf,
        "btl": float((ade < lin).mean()) if lin.notna().any() else -np.inf,
        "mean_ade": float(ade.mean()) if ade.notna().any() else np.inf,
        "switch_rate": float(sw.mean()) if sw.notna().any() else 0.0,
        "n": int(sel["task_uid"].nunique()) if "task_uid" in sel.columns else int(len(sel)),
    }


def make_v26a_safety_gated_v25h_selector(
    metrics: pd.DataFrame,
    validation_split: str = "validation",
    selector_name: str = V26A_SELECTOR_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V26a: V25h weighted-K20 candidate strength with a learned safety gate.

    The old V25h weighted representative paths had the best median ADE but weak
    outlier behavior.  This selector keeps robust_global/linear as a backbone and
    switches to weighted/residual candidates only when a validation-trained gate
    predicts a clear improvement over the safe default.
    """
    if metrics is None or metrics.empty or "task_uid" not in metrics.columns or "method" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    mm = add_linear_baseline_metrics(metrics.copy())
    default_method = _v26_default_method(mm)
    pool_methods = _v26_v25h_weighted_candidate_methods(mm, include_safe=True)
    if default_method not in pool_methods:
        pool_methods.append(default_method)
    pool = mm[mm["method"].astype(str).isin(pool_methods)].copy()
    pool = _v26_attach_default(pool, default_method)
    if pool.empty:
        return pd.DataFrame(), pd.DataFrame()
    val = pool[pool.get("split", pd.Series(index=pool.index, dtype=str)).astype(str).eq(validation_split)].copy()
    train = val[~val["method"].astype(str).eq(default_method)].copy()
    if train.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Positive = candidate beats the safe default by a small but meaningful paired margin.
    y_raw = pd.to_numeric(train["ADE"], errors="coerce") < (pd.to_numeric(train["v26_default_ADE"], errors="coerce") * 0.985)
    train = train[y_raw.notna()].copy()
    y = y_raw.loc[train.index].astype(int).to_numpy()
    X_train, cols = _v26_safe_feature_frame(train, train_columns=None)
    method_priors = train.assign(_y=y).groupby("method")['_y'].mean().to_dict()
    model, model_name = _v26_fit_classifier(X_train, y)
    pool["v26a_safety_probability"] = _v26_predict_probability(pool, model, cols, method_priors)
    # Validation method-level prior prevents a single high-probability outlier method from dominating.
    val_prior = val.groupby("method").agg(
        v26a_val_method_ratio=("ade_ratio_to_linear", lambda s: float(pd.to_numeric(s, errors="coerce").median())),
        v26a_val_method_btl=("better_than_linear", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
    ).reset_index()
    pool = pool.merge(val_prior, on="method", how="left")
    pool["v26a_val_method_ratio"] = pd.to_numeric(pool.get("v26a_val_method_ratio"), errors="coerce").fillna(1.5)
    pool["v26a_val_method_btl"] = pd.to_numeric(pool.get("v26a_val_method_btl"), errors="coerce").fillna(0.0)

    risky_methods = set(_v26_v25h_weighted_candidate_methods(mm, include_safe=False))

    def select_with_threshold(source: pd.DataFrame, threshold: float, max_val_ratio: float) -> pd.DataFrame:
        rows = []
        for uid, g in source.groupby("task_uid", sort=False):
            gg = g.copy()
            default_g = gg[gg["method"].astype(str).eq(default_method)]
            default_row = default_g.iloc[0].copy() if not default_g.empty else gg.iloc[0].copy()
            cand = gg[gg["method"].astype(str).isin(risky_methods)].copy()
            cand = cand[pd.to_numeric(cand.get("v26a_safety_probability", np.nan), errors="coerce").ge(float(threshold))]
            cand = cand[pd.to_numeric(cand.get("v26a_val_method_ratio", np.nan), errors="coerce").le(float(max_val_ratio))]
            if not cand.empty:
                cand["_prob"] = pd.to_numeric(cand["v26a_safety_probability"], errors="coerce")
                cand["_prior"] = pd.to_numeric(cand["v26a_val_method_ratio"], errors="coerce")
                cand["_btl"] = pd.to_numeric(cand["v26a_val_method_btl"], errors="coerce")
                chosen = cand.sort_values(["_prob", "_prior", "_btl", "method"], ascending=[False, True, False, True], kind="mergesort").iloc[0].copy()
                used = 1
            else:
                chosen = default_row.copy()
                used = 0
            source_method = str(chosen.get("method", "unknown"))
            chosen["method"] = selector_name
            chosen["source_method"] = source_method
            chosen["v26_selected_source_method"] = source_method
            chosen["selector_used_risky"] = int(used)
            chosen["selector_default_method"] = default_method
            chosen["selector_probability_threshold"] = float(threshold)
            chosen["selector_max_validation_method_ratio"] = float(max_val_ratio)
            chosen["selector_version"] = "v26a_v25h_weighted_safety_gate"
            rows.append(chosen.to_dict())
        return pd.DataFrame(rows)

    val_pool = pool[pool.get("split", pd.Series(index=pool.index, dtype=str)).astype(str).eq(validation_split)].copy()
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    max_ratios = [0.98, 1.00, 1.03, 1.06, 1.10, 1.20]
    scored = []
    for th in thresholds:
        for mr in max_ratios:
            rows = select_with_threshold(val_pool, th, mr)
            score = _v26_selection_score(mm, rows, selector_name, split=validation_split)
            scored.append({"threshold": float(th), "max_val_ratio": float(mr), **score})
    ss = pd.DataFrame(scored).sort_values(
        ["med_ratio", "mean_ratio", "q90_ratio", "btl", "switch_rate"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    best = ss.iloc[0]
    out = select_with_threshold(pool, float(best["threshold"]), float(best["max_val_ratio"]))
    out = add_linear_baseline_metrics(pd.concat([mm[mm["method"].astype(str).eq("linear")], out], ignore_index=True, sort=False))
    out = out[out["method"].astype(str).eq(selector_name)].copy()
    choice = pd.DataFrame([{
        "paper_method": selector_name,
        "selected_method": "task_specific_v25h_weighted_candidate_with_safety_gate",
        "default_method": default_method,
        "candidate_pool_n_methods": int(len(pool_methods)),
        "selector_model": model_name,
        "selector_threshold": float(best["threshold"]),
        "selector_max_validation_method_ratio": float(best["max_val_ratio"]),
        "validation_ADE_ratio_median": float(best["med_ratio"]),
        "validation_ADE_ratio_mean": float(best["mean_ratio"]),
        "validation_q90_ratio": float(best["q90_ratio"]),
        "validation_better_than_linear_rate": float(best["btl"]),
        "validation_switch_rate": float(best["switch_rate"]),
        "selection_reason": "v26a_borrow_v25h_weighted_k20_with_learned_safety_gate",
    }])
    return out, choice


def _v26b_regime_label_from_row(row: pd.Series) -> str:
    dataset = str(row.get("dataset", "")).lower()
    taxon = str(row.get("taxon", "")).lower()
    setting = str(row.get("setting_name", "")).lower()
    species = str(row.get("species_id", row.get("species_common_name", ""))).lower()
    text = " ".join([dataset, taxon, setting, species])
    if "bobcat" in text:
        return "olympic_bobcat_240_to_60"
    if "cougar" in text:
        return "olympic_cougar_240_to_60"
    if "puma" in text:
        if "240" in setting:
            return "puma_long_240_to_5"
        if "120" in setting:
            return "puma_mid_120_to_5"
        if "60" in setting:
            return "puma_short_60_to_5"
        if "30" in setting:
            return "puma_short_30_to_5"
        return "puma_other"
    if "tiger" in text or "leopard" in text or "thailand" in text:
        species_part = "leopard" if "leopard" in text else ("tiger" if "tiger" in text else taxon or "panthera")
        if "60" in setting and "15" in setting:
            return f"thailand_{species_part}_60_to_15"
        if "240" in setting and "60" in setting:
            return f"thailand_{species_part}_240_to_60"
        return f"thailand_{species_part}_other"
    return "|".join([str(row.get(c, "missing")) for c in ["dataset", "taxon", "setting_name"]])


def _v26b_choose_method_for_regime(val: pd.DataFrame, candidate_methods: list[str], default_method: str) -> tuple[str, dict]:
    g = val[val["method"].astype(str).isin(candidate_methods)].copy()
    if g.empty:
        return default_method, {"reason": "empty_regime_pool"}
    rows = []
    default_g = g[g["method"].astype(str).eq(default_method)].copy()
    default_med = float(pd.to_numeric(default_g.get("ade_ratio_to_linear", pd.Series(dtype=float)), errors="coerce").median()) if not default_g.empty else 1.0
    default_mean = float(pd.to_numeric(default_g.get("ADE", pd.Series(dtype=float)), errors="coerce").mean()) if not default_g.empty else np.inf
    default_q90 = float(pd.to_numeric(default_g.get("ade_ratio_to_linear", pd.Series(dtype=float)), errors="coerce").quantile(0.90)) if not default_g.empty else np.inf
    default_btl = float(pd.to_numeric(default_g.get("better_than_linear", pd.Series(dtype=float)), errors="coerce").mean()) if not default_g.empty else 0.0
    for method, m in g.groupby("method", sort=False):
        ade = pd.to_numeric(m.get("ADE", pd.Series(index=m.index, dtype=float)), errors="coerce")
        ratio = pd.to_numeric(m.get("ade_ratio_to_linear", pd.Series(index=m.index, dtype=float)), errors="coerce")
        btl = pd.to_numeric(m.get("better_than_linear", pd.Series(index=m.index, dtype=float)), errors="coerce")
        rows.append({
            "method": str(method),
            "n_tasks": int(m["task_uid"].nunique()),
            "med_ratio": float(ratio.median()) if ratio.notna().any() else np.inf,
            "mean_ade": float(ade.mean()) if ade.notna().any() else np.inf,
            "q90_ratio": float(ratio.quantile(0.90)) if ratio.notna().any() else np.inf,
            "btl": float(btl.mean()) if btl.notna().any() else 0.0,
        })
    s = pd.DataFrame(rows)
    if s.empty:
        return default_method, {"reason": "no_summary"}
    # Conservative eligibility: median must improve or match default, and tail/mean cannot degrade too much.
    eligible = s[
        (s["n_tasks"].ge(2))
        & (s["med_ratio"].le(min(default_med - 0.005, 0.995) if np.isfinite(default_med) else 0.995))
        & (s["mean_ade"].le(default_mean + 10.0 if np.isfinite(default_mean) else np.inf))
        & (s["q90_ratio"].le(default_q90 + 0.20 if np.isfinite(default_q90) else np.inf))
        & (s["btl"].ge(max(0.30, default_btl - 0.10)))
    ].copy()
    if eligible.empty:
        # If nothing passes conservative guard, allow linear fallback if it is the validation winner.
        s2 = s.sort_values(["med_ratio", "mean_ade", "q90_ratio", "btl"], ascending=[True, True, True, False], kind="mergesort")
        best_method = str(s2.iloc[0]["method"])
        if best_method in {"linear", "heading_hermite"} or float(s2.iloc[0]["med_ratio"]) <= default_med - 0.03:
            row = s2.iloc[0].to_dict(); row["reason"] = "fallback_to_clear_regime_winner"
            return best_method, row
        return default_method, {"reason": "default_guarded_no_eligible", "default_med_ratio": default_med, "default_btl": default_btl}
    eligible = eligible.sort_values(["med_ratio", "mean_ade", "q90_ratio", "btl", "method"], ascending=[True, True, True, False, True], kind="mergesort")
    row = eligible.iloc[0].to_dict(); row["reason"] = "regime_validation_selected"
    return str(row["method"]), row


def make_v26b_regime_aware_selector(
    metrics: pd.DataFrame,
    validation_split: str = "validation",
    selector_name: str = V26B_SELECTOR_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V26b: regime-aware selector using validation within movement regimes.

    One global selector mixed puma short gaps, puma long gaps, bobcat/cougar, and
    Thailand Panthera tasks.  V26b chooses a method per interpretable regime,
    with conservative default fallback when validation does not support switching.
    """
    if metrics is None or metrics.empty or "task_uid" not in metrics.columns or "method" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    mm = add_linear_baseline_metrics(metrics.copy())
    default_method = _v26_default_method(mm)
    candidate_methods = _v26_v25h_weighted_candidate_methods(mm, include_safe=True)
    for m in ["linear", "heading_hermite", default_method]:
        if m in _v26_method_values(mm) and m not in candidate_methods:
            candidate_methods.append(m)
    pool = mm[mm["method"].astype(str).isin(candidate_methods)].copy()
    if pool.empty:
        return pd.DataFrame(), pd.DataFrame()
    pool["v26b_regime"] = pool.apply(_v26b_regime_label_from_row, axis=1)
    val = pool[pool.get("split", pd.Series(index=pool.index, dtype=str)).astype(str).eq(validation_split)].copy()
    if val.empty:
        return pd.DataFrame(), pd.DataFrame()
    method_by_regime: dict[str, str] = {}
    choice_rows = []
    for regime, vg in val.groupby("v26b_regime", sort=False):
        selected, info = _v26b_choose_method_for_regime(vg, candidate_methods, default_method)
        method_by_regime[str(regime)] = selected
        choice = {
            "paper_method": selector_name,
            "regime": str(regime),
            "selected_method": selected,
            "default_method": default_method,
            "validation_n_tasks": int(vg["task_uid"].nunique()),
            "selection_reason": info.get("reason", "unknown"),
        }
        for k, v in info.items():
            if k != "method":
                choice[f"validation_{k}"] = v
        choice_rows.append(choice)
    rows = []
    for uid, g in pool.groupby("task_uid", sort=False):
        gg = g.copy()
        regime = str(gg["v26b_regime"].iloc[0])
        selected = method_by_regime.get(regime, default_method)
        src = gg[gg["method"].astype(str).eq(selected)].copy()
        if src.empty:
            src = gg[gg["method"].astype(str).eq(default_method)].copy()
            selected = default_method
        if src.empty:
            src = gg[gg["method"].astype(str).eq("linear")].copy()
            selected = "linear"
        if src.empty:
            src = gg.iloc[[0]].copy()
            selected = str(src.iloc[0].get("method", "unknown"))
        rr = src.iloc[0].copy()
        rr["method"] = selector_name
        rr["source_method"] = selected
        rr["v26_selected_source_method"] = selected
        rr["v26b_regime"] = regime
        rr["selector_default_method"] = default_method
        rr["selector_version"] = "v26b_regime_aware_validation_selector"
        rows.append(rr.to_dict())
    out = pd.DataFrame(rows)
    out = add_linear_baseline_metrics(pd.concat([mm[mm["method"].astype(str).eq("linear")], out], ignore_index=True, sort=False))
    out = out[out["method"].astype(str).eq(selector_name)].copy()
    choice_df = pd.DataFrame(choice_rows)
    return out, choice_df


def _v26c_rank_methods_by_validation(metrics: pd.DataFrame, candidate_methods: list[str], validation_split: str) -> dict[str, list[str]]:
    pool = add_linear_baseline_metrics(metrics.copy())
    pool["v26c_regime"] = pool.apply(_v26b_regime_label_from_row, axis=1)
    val = pool[pool.get("split", pd.Series(index=pool.index, dtype=str)).astype(str).eq(validation_split)].copy()
    ranks: dict[str, list[str]] = {}
    if val.empty:
        ranks["__global__"] = candidate_methods
        return ranks
    for regime, g in val.groupby("v26c_regime", sort=False):
        gg = g[g["method"].astype(str).isin(candidate_methods)].copy()
        rows = []
        for method, m in gg.groupby("method", sort=False):
            ratio = pd.to_numeric(m.get("ade_ratio_to_linear", pd.Series(index=m.index, dtype=float)), errors="coerce")
            ade = pd.to_numeric(m.get("ADE", pd.Series(index=m.index, dtype=float)), errors="coerce")
            btl = pd.to_numeric(m.get("better_than_linear", pd.Series(index=m.index, dtype=float)), errors="coerce")
            rows.append({
                "method": str(method),
                "med_ratio": float(ratio.median()) if ratio.notna().any() else np.inf,
                "mean_ade": float(ade.mean()) if ade.notna().any() else np.inf,
                "q90_ratio": float(ratio.quantile(0.90)) if ratio.notna().any() else np.inf,
                "btl": float(btl.mean()) if btl.notna().any() else 0.0,
            })
        ss = pd.DataFrame(rows)
        if ss.empty:
            ranks[str(regime)] = candidate_methods
        else:
            ss = ss.sort_values(["med_ratio", "mean_ade", "q90_ratio", "btl", "method"], ascending=[True, True, True, False, True], kind="mergesort")
            ranks[str(regime)] = list(ss["method"].astype(str))
    return ranks


def make_v26c_topk_calibrated_set_diagnostics(
    metrics: pd.DataFrame,
    validation_split: str = "validation",
    selector_prefix: str = V26C_PREFIX,
    ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V26c: Top-K diagnostic rows for candidate-set/uncertainty evaluation.

    For each validation-defined regime, candidate method families are ranked by
    validation performance.  For each task, the diagnostic row reports the best
    path *inside the Top-K set*.  This is an oracle-within-set uncertainty metric,
    not a deployable Top-1 selector; it should be reported as Top-K coverage.
    """
    if metrics is None or metrics.empty or "task_uid" not in metrics.columns or "method" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    mm = add_linear_baseline_metrics(metrics.copy())
    candidate_methods = _v26_proposed_nonselector_methods(mm, include_baseline_fallbacks=False)
    if not candidate_methods:
        return pd.DataFrame(), pd.DataFrame()
    ranks = _v26c_rank_methods_by_validation(mm, candidate_methods, validation_split=validation_split)
    mm["v26c_regime"] = mm.apply(_v26b_regime_label_from_row, axis=1)
    rows = []
    choice_rows = []
    for k in ks:
        method_name = f"{selector_prefix}_top{k}_validation_ranked_set_oracle"
        for uid, g in mm[mm["method"].astype(str).isin(candidate_methods)].groupby("task_uid", sort=False):
            regime = str(g["v26c_regime"].iloc[0])
            method_rank = ranks.get(regime, ranks.get("__global__", candidate_methods))
            allowed = method_rank[:min(int(k), len(method_rank))]
            gg = g[g["method"].astype(str).isin(allowed)].copy()
            if gg.empty:
                continue
            gg["_ade"] = pd.to_numeric(gg["ADE"], errors="coerce")
            best = gg.sort_values(["_ade", "method"], ascending=[True, True], kind="mergesort").iloc[0].copy()
            src_method = str(best.get("method", "unknown"))
            best["method"] = method_name
            best["source_method"] = src_method
            best["v26c_topk_k"] = int(k)
            best["v26c_regime"] = regime
            best["v26c_topk_methods"] = ";".join(allowed)
            best["v26c_oracle_within_set"] = 1
            best["v26c_not_deployable_top1"] = 1
            best["selector_version"] = "v26c_validation_ranked_topk_set_diagnostic"
            rows.append(best.to_dict())
        for regime, method_rank in ranks.items():
            choice_rows.append({
                "paper_method": method_name,
                "regime": regime,
                "topk_k": int(k),
                "ranked_methods": ";".join(method_rank[:min(int(k), len(method_rank))]),
                "selection_reason": "v26c_topk_uncertainty_set_ranked_by_validation_regime",
            })
    # Full proposed-pool oracle diagnostic.
    full_name = f"{selector_prefix}_full_proposed_pool_oracle"
    for uid, g in mm[mm["method"].astype(str).isin(candidate_methods)].groupby("task_uid", sort=False):
        gg = g.copy()
        gg["_ade"] = pd.to_numeric(gg["ADE"], errors="coerce")
        best = gg.sort_values(["_ade", "method"], ascending=[True, True], kind="mergesort").iloc[0].copy()
        src_method = str(best.get("method", "unknown"))
        best["method"] = full_name
        best["source_method"] = src_method
        best["v26c_topk_k"] = int(len(candidate_methods))
        best["v26c_topk_methods"] = "ALL_PROPOSED_NONSELECTOR_METHODS"
        best["v26c_oracle_within_set"] = 1
        best["v26c_not_deployable_top1"] = 1
        best["selector_version"] = "v26c_full_pool_oracle_diagnostic"
        rows.append(best.to_dict())
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_linear_baseline_metrics(pd.concat([mm[mm["method"].astype(str).eq("linear")], out], ignore_index=True, sort=False))
        out = out[out["method"].astype(str).str.startswith(selector_prefix)].copy()
    choice_df = pd.DataFrame(choice_rows)
    return out, choice_df


# -----------------------------------------------------------------------------
# final paired LR->HR super-resolution generation experiments
# -----------------------------------------------------------------------------
# final moves part of the intelligence earlier in the pipeline.  Instead of only
# generating a broad candidate bank and selecting after evaluation, it adds new
# generation-layer paths that reuse matched low-resolution/high-resolution
# motifs from the training library:
#   finala: rank the retrieved LR->HR paths directly and output a small SR bank;
#   finalb: recover high-resolution residual structure as a sparse coefficient
#         combination of a few matched HR motifs;
#   diverse SR: build a deliberately diverse Top-10 SR set and report Top-K set
#         diagnostics.
# These methods use only truth-free retrieval/scoring information.  Truth ADE is
# used later only for validation/test reporting, never to generate the path.

final_ACTIVE_VARIANT = "c"  # one of: a, b, c
finalA_PREFIX = "pretrained_motif_v27a_lrhr_sr"
finalB_PREFIX = "pretrained_motif_v27b_sparsecoef_sr"
finalC_PREFIX = "probabilistic_tg_diverse_sr"


def _v27_score_label(score_col: str) -> str:
    return _score_label(score_col) if '_score_label' in globals() else str(score_col).replace('_score', '')


def _v27_primary_score_frame(proposed: pd.DataFrame, candidate_paths: dict[str, np.ndarray], score_col: str = "cost_rank_score") -> pd.DataFrame:
    """Build a truth-free SR ranking table from retrieved LR->HR motifs only."""
    if proposed is None or proposed.empty:
        return pd.DataFrame()
    x = _retrieved_candidates(proposed)
    if x is None or x.empty:
        x = proposed.copy()
    if x.empty or "candidate_id" not in x.columns:
        return pd.DataFrame()
    x = x.copy()
    # Keep only candidates with an actual generated path.
    x["_candidate_id_str"] = x["candidate_id"].astype(str)
    x = x[x["_candidate_id_str"].map(lambda cid: cid in candidate_paths)].copy()
    if x.empty:
        return pd.DataFrame()
    if score_col not in x.columns:
        score_col = "cost_rank_score" if "cost_rank_score" in x.columns else ("proposal_score" if "proposal_score" in x.columns else None)
    base_score = _num(x[score_col], default=0.0) if score_col is not None else pd.Series(0.0, index=x.index)
    # The existing cost-rank score remains dominant.  The small penalties below
    # gently prefer LR->HR matches with compatible direction/context/shape and
    # avoid very extreme path ratios.  Scales are intentionally mild so this does
    # not become another overfit selector.
    sr_score = base_score.astype(float).copy()
    for col, weight in [
        ("candidate_cost_source_shape", 0.08),
        ("candidate_cost_direction", 0.06),
        ("candidate_cost_efficiency", 0.05),
        ("context_cost", 0.04),
        ("step_violation_fraction", 0.20),
        ("step_q90_over_capacity", 0.05),
    ]:
        if col in x.columns:
            sr_score = sr_score - float(weight) * _num(x[col], default=0.0).astype(float)
    if "path_ratio" in x.columns:
        r = _num(x["path_ratio"], default=np.nan).astype(float)
        ratio_pen = np.abs(np.log(np.clip(r.to_numpy(dtype=float), 1e-6, 1e6)))
        sr_score = sr_score - 0.03 * pd.Series(np.nan_to_num(ratio_pen, nan=0.0), index=x.index)
    if "listwise_probability" in x.columns:
        p = _num(x["listwise_probability"], default=0.0).astype(float)
        sr_score = sr_score + 0.10 * p
    x["sr_score"] = pd.to_numeric(sr_score, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-1e9)
    if "candidate_order" not in x.columns:
        x["candidate_order"] = np.arange(len(x))
    x = x.sort_values(["sr_score", "candidate_order"], ascending=[False, True], kind="mergesort")
    return x


def _v27_mean_path_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        aa = np.asarray(a, dtype=float); bb = np.asarray(b, dtype=float)
        n = min(len(aa), len(bb))
        if n <= 0 or aa.ndim != 2 or bb.ndim != 2:
            return 0.0
        return float(np.nanmean(np.linalg.norm(aa[:n, :2] - bb[:n, :2], axis=1)))
    except Exception:
        return 0.0


def _v27_diversity_threshold(path: np.ndarray) -> float:
    p = np.asarray(path, dtype=float)
    if p.ndim != 2 or len(p) < 2:
        return 5.0
    disp = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
    return float(np.clip(0.08 * disp, 5.0, 75.0))


def _v27_select_diverse_rows(rows: pd.DataFrame, candidate_paths: dict[str, np.ndarray], k: int = 10, diversity_scale: float = 1.0) -> pd.DataFrame:
    """Select a compact diverse set in SR-rank order; fill with next-best if needed."""
    if rows is None or rows.empty:
        return pd.DataFrame()
    selected = []
    selected_ids = set()
    for _, row in rows.iterrows():
        cid = str(row.get("candidate_id", row.get("_candidate_id_str", "")))
        path = candidate_paths.get(cid)
        if path is None:
            continue
        thr = _v27_diversity_threshold(path) * float(diversity_scale)
        keep = True
        for sid in selected_ids:
            sp = candidate_paths.get(sid)
            if sp is not None and _v27_mean_path_distance(path, sp) < thr:
                keep = False; break
        if keep:
            selected.append(row); selected_ids.add(cid)
        if len(selected) >= int(k):
            break
    if len(selected) < int(k):
        for _, row in rows.iterrows():
            cid = str(row.get("candidate_id", row.get("_candidate_id_str", "")))
            if cid in selected_ids or cid not in candidate_paths:
                continue
            selected.append(row); selected_ids.add(cid)
            if len(selected) >= int(k):
                break
    return pd.DataFrame(selected)


def _v27_common_meta(row: pd.Series, rank: int, score_col: str, method: str, origin: str) -> dict:
    meta = {
        "candidate_id": str(row.get("candidate_id", row.get("_candidate_id_str", ""))),
        "candidate_origin": origin,
        "style": str(row.get("style", "retrieved_motif")),
        "score_col": score_col,
        "path_topK": int(rank),
        "path_weight_beta": np.nan,
        "n_source_candidates": 1,
        "n_retrieved_source_candidates": 1,
        "used_endpoint_fallback_only": 0,
        "source_method": method,
        "method_family_detail": origin,
        "sr_active_variant": final_ACTIVE_VARIANT,
        "sr_rank": int(rank),
        "sr_score": float(row.get("sr_score", np.nan)) if pd.notna(row.get("sr_score", np.nan)) else np.nan,
        **_source_relation_summary(pd.DataFrame([row])),
    }
    meta.update(_candidate_diagnostics_from_row(row))
    return meta


def _v27a_lrhr_retrieval_paths(task: ReconstructionTask, proposed: pd.DataFrame, candidate_paths: dict[str, np.ndarray], k: int = 10) -> dict[str, tuple[np.ndarray, dict]]:
    """finala: output a small bank of directly retrieved LR->HR paths ranked by SR score."""
    out: dict[str, tuple[np.ndarray, dict]] = {}
    rows = _v27_primary_score_frame(proposed, candidate_paths, score_col="cost_rank_score")
    rows = _v27_select_diverse_rows(rows, candidate_paths, k=k, diversity_scale=0.65)
    if rows.empty:
        return out
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        cid = str(row.get("candidate_id", row.get("_candidate_id_str", "")))
        path = candidate_paths.get(cid)
        if path is None:
            continue
        method = f"{finalA_PREFIX}_rank{rank:02d}"
        meta = _v27_common_meta(row, rank, "cost_rank_score", method, "paired_lrhr_retrieval_super_resolution")
        meta.update({
            "sr_generation_mode": "ranked_lrhr_retrieval",
            "sr_topk_pool_size": int(k),
            "sr_is_direct_generation": 1,
        })
        out[method] = (np.asarray(path, dtype=float), meta)
    return out


def _v27_sparse_residual_path(task: ReconstructionTask, rows: pd.DataFrame, candidate_paths: dict[str, np.ndarray], k: int = 5, beta: float = 1.25, base_name: str = "linear") -> tuple[np.ndarray | None, dict]:
    if rows is None or rows.empty:
        return None, {}
    selected = rows.head(int(k)).copy()
    paths = []
    used_rows = []
    for _, row in selected.iterrows():
        cid = str(row.get("candidate_id", row.get("_candidate_id_str", "")))
        p = candidate_paths.get(cid)
        if p is not None:
            paths.append(np.asarray(p, dtype=float)); used_rows.append(row)
    if not paths:
        return None, {}
    baselines = generate_baseline_paths(task)
    base = baselines.get(base_name, None)
    if base is None:
        base = baselines.get("linear", None)
    if base is None:
        return None, {}
    base = np.asarray(base, dtype=float)
    n = min([len(base)] + [len(p) for p in paths])
    if n <= 1:
        return None, {}
    base = base[:n]
    # Compute residuals relative to each motif's own straight connector.  This
    # transfers only high-resolution shape detail, not source segment endpoints.
    residuals = []
    scores = []
    for p, row in zip(paths, used_rows):
        p = p[:n]
        motif_linear = np.column_stack([
            np.linspace(p[0, 0], p[-1, 0], n),
            np.linspace(p[0, 1], p[-1, 1], n),
        ])
        residuals.append(p - motif_linear)
        scores.append(float(row.get("sr_score", 0.0)))
    scores = np.asarray(scores, dtype=float)
    w = _softmax_score(scores, beta=float(beta))
    resid = np.zeros_like(base, dtype=float)
    for wi, ri in zip(w, residuals):
        resid += float(wi) * ri
    conf = _candidate_confidence_from_rows(pd.DataFrame(used_rows))
    lam = float(np.clip(0.35 + 0.55 * conf, 0.20, 0.92))
    xy = base + lam * resid
    xy[0] = base[0]; xy[-1] = base[-1]
    meta = _weighted_candidate_diagnostics(pd.DataFrame(used_rows), w, prefix="v27_sparse_")
    meta.update(_source_relation_summary(pd.DataFrame(used_rows)))
    meta.update({
        "candidate_id": ";".join(str(r.get("candidate_id", r.get("_candidate_id_str", ""))) for r in used_rows),
        "candidate_origin": "paired_lrhr_sparse_residual_super_resolution",
        "style": "sparse_lrhr_residual_combination",
        "score_col": "sr_score",
        "path_topK": int(k),
        "path_weight_beta": float(beta),
        "n_source_candidates": int(len(used_rows)),
        "n_retrieved_source_candidates": int(len(used_rows)),
        "used_endpoint_fallback_only": 0,
        "source_method": f"sparse_residual_{base_name}_K{k}_b{beta:g}",
        "method_family_detail": "sparse_residual_lrhr_super_resolution",
        "sr_generation_mode": "sparse_coefficients_over_hr_residuals",
        "sr_active_variant": final_ACTIVE_VARIANT,
        "v27_sparse_base": base_name,
        "v27_sparse_k": int(k),
        "v27_sparse_beta": float(beta),
        "v27_sparse_lambda": float(lam),
        "selector_confidence": float(conf),
        "adaptive_confidence": float(conf),
        "v27_sparse_weights": ";".join(f"{float(x):.4f}" for x in w),
    })
    return xy, meta


def _v27b_sparsecoef_paths(task: ReconstructionTask, proposed: pd.DataFrame, candidate_paths: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, dict]]:
    """finalb: generate HR detail from sparse coefficients over matched HR residuals."""
    out: dict[str, tuple[np.ndarray, dict]] = {}
    rows = _v27_primary_score_frame(proposed, candidate_paths, score_col="cost_rank_score")
    # Use a compact high-quality set rather than dense averaging over 20-40 paths.
    rows = _v27_select_diverse_rows(rows, candidate_paths, k=12, diversity_scale=0.50)
    if rows.empty:
        return out
    for base_name in ["linear", "heading_hermite"]:
        for k, beta in [(3, 1.50), (5, 1.25), (8, 1.00)]:
            xy, meta = _v27_sparse_residual_path(task, rows, candidate_paths, k=k, beta=beta, base_name=base_name)
            if xy is None:
                continue
            method = f"{finalB_PREFIX}_{base_name}_K{k}_b{str(beta).replace('.', 'p')}"
            meta = dict(meta)
            meta["source_method"] = method
            meta["sr_is_direct_generation"] = 1
            out[method] = (xy, meta)
    return out


def _v27c_diverse_sr_paths(task: ReconstructionTask, proposed: pd.DataFrame, candidate_paths: dict[str, np.ndarray], k: int = 10) -> dict[str, tuple[np.ndarray, dict]]:
    """diverse SR: generate an explicitly diverse Top-10 SR set for uncertainty analysis."""
    out: dict[str, tuple[np.ndarray, dict]] = {}
    rows = _v27_primary_score_frame(proposed, candidate_paths, score_col="cost_rank_score")
    rows = _v27_select_diverse_rows(rows, candidate_paths, k=k, diversity_scale=1.15)
    if rows.empty:
        return out
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        cid = str(row.get("candidate_id", row.get("_candidate_id_str", "")))
        path = candidate_paths.get(cid)
        if path is None:
            continue
        method = f"{finalC_PREFIX}_rank{rank:02d}"
        meta = _v27_common_meta(row, rank, "cost_rank_score", method, "diverse_paired_lrhr_topk_super_resolution")
        meta.update({
            "sr_generation_mode": "diverse_topk_lrhr_retrieval",
            "sr_topk_pool_size": int(k),
            "sr_diverse_rank": int(rank),
            "sr_is_direct_generation": 1,
        })
        out[method] = (np.asarray(path, dtype=float), meta)
    return out


_previous_proposed_paths_for_task_v27base = proposed_paths_for_task


def proposed_paths_for_task(
    task: ReconstructionTask,
    scored: pd.DataFrame,
    candidate_paths: dict[str, np.ndarray],
    config: ReconstructionConfig,
    betas: Sequence[float] = REPRESENTATIVE_BETAS,
) -> dict[str, tuple[np.ndarray, dict]]:
    """Return the retained publication candidate bank for one task.

    Retained families:
    - motif retrieval Top-1 and weighted representatives used by robust Top-1;
    - diverse motif super-resolution ranks used by the balanced Top-10 set.

    Older residual-flow/adaptive selector experiments are intentionally omitted
    from the publication benchmark for speed and clarity.
    """
    out = _previous_proposed_paths_for_task_v206(task, scored, candidate_paths, config, betas=betas)
    proposed = _proposed_candidates(scored)
    try:
        out.update(_v27c_diverse_sr_paths(task, proposed, candidate_paths, k=10))
    except Exception as exc:
        status(f"Diverse motif candidate generation skipped for task {getattr(task, 'task_uid', 'unknown')} after error: {exc}")
    return out


def make_v27c_diverse_topk_diagnostics(metrics: pd.DataFrame, ks: Sequence[int] = (1, 3, 5, 10), selector_prefix: str = finalC_PREFIX) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Oracle-within-generated-set diagnostics for the diverse super-resolution Top-K candidate set.

    These rows are not deployable Top-1 selectors.  They answer: if we output the
    first K diverse SR trajectories, how close is the closest one to truth?
    """
    if metrics is None or metrics.empty or "task_uid" not in metrics.columns or "method" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    mm = add_linear_baseline_metrics(metrics.copy())
    methods = [m for m in sorted(mm["method"].dropna().astype(str).unique()) if m.startswith(f"{selector_prefix}_rank")]
    if not methods:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    choice_rows = []
    for k in ks:
        allowed = [m for m in methods if int(str(m).split("rank")[-1]) <= int(k)]
        if not allowed:
            continue
        method_name = f"{selector_prefix}_top{k}_generated_set_oracle"
        for uid, g in mm[mm["method"].astype(str).isin(allowed)].groupby("task_uid", sort=False):
            gg = g.copy()
            gg["_ade"] = pd.to_numeric(gg["ADE"], errors="coerce")
            best = gg.sort_values(["_ade", "method"], ascending=[True, True], kind="mergesort").iloc[0].copy()
            src = str(best.get("method", "unknown"))
            best["method"] = method_name
            best["source_method"] = src
            best["candidate_set_k"] = int(k)
            best["candidate_set_methods"] = ";".join(allowed)
            best["oracle_within_generated_set"] = 1
            best["not_deployable_top1"] = 1
            best["selector_version"] = "diverse_sr_candidate_set_diagnostic"
            rows.append(best.to_dict())
        choice_rows.append({
            "paper_method": method_name,
            "topk_k": int(k),
            "ranked_methods": ";".join(allowed),
            "selection_reason": "oracle_within_generated_diverse_sr_set",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_linear_baseline_metrics(pd.concat([mm[mm["method"].astype(str).eq("linear")], out], ignore_index=True, sort=False))
        out = out[out["method"].astype(str).str.startswith(selector_prefix)].copy()
    return out, pd.DataFrame(choice_rows)


# -----------------------------------------------------------------------------
# Paper-facing reporting helpers for the final package
# -----------------------------------------------------------------------------
PAPER_METHOD_ALIASES = {
    "linear": "linear_interpolation",
    "heading_hermite": "heading_aware_hermite",
    "brownian_bridge": "brownian_bridge",
    "rtg_bridge": "random_time_geographic_bridge",
    "pretrained_motif_guarded": "guarded_pretrained_top1",
    "pretrained_motif_robust_global": "robust_pretrained_top1",
    "pretrained_motif_conservative_oracle_switcher": "conservative_pretrained_top1",
    "balanced_pooled_extension_top10_minADE_candidate_set": "balanced_pooled_top10_candidate_set",
}


def paper_method_label(method: str) -> str:
    """Return clean, publishable method labels for paper-facing tables."""
    m = str(method)
    if m in PAPER_METHOD_ALIASES:
        return PAPER_METHOD_ALIASES[m]
    if m.startswith("probabilistic_tg_diverse_sr_rank"):
        rank = m.split("rank")[-1]
        return f"diverse_sr_candidate_rank{rank}"
    if m.startswith("probabilistic_tg_diverse_sr_top"):
        # probabilistic_tg_diverse_sr_top10_generated_set_oracle
        mm = re.search(r"_top(\d+)_generated_set_oracle", m)
        if mm:
            return f"diverse_sr_top{mm.group(1)}_minADE_candidate_set"
    if m.startswith("expanded_sr_top"):
        return m
    if m == "balanced_pooled_extension_top10_minADE_candidate_set":
        return "balanced_pooled_top10_candidate_set"
    return m


def paper_metrics_for_reporting(metrics: pd.DataFrame, include_generated_ranks: bool = False) -> pd.DataFrame:
    """Filter and relabel metrics for paper/dissertation reporting.

    This keeps baselines, the robust deployable Top-1 method, and
    the balanced pooled Top-10 candidate set. Full diagnostics remain saved
    separately by the notebooks.
    """
    if metrics is None or metrics.empty or "method" not in metrics.columns:
        return pd.DataFrame() if metrics is None else metrics.copy()
    method = metrics["method"].astype(str)
    keep = method.isin({
        "linear", "heading_hermite", "brownian_bridge", "rtg_bridge",
        "pretrained_motif_robust_global",
        "balanced_pooled_extension_top10_minADE_candidate_set",
    })
    # Keep older compact candidate-set diagnostics in full audit files only; the
    # paper-facing table focuses on robust Top-1 and balanced Top-10.
    if include_generated_ranks:
        keep |= method.str.match(r"^probabilistic_tg_diverse_sr_rank\d+$")
    out = metrics.loc[keep].copy()
    out["method_internal"] = out["method"].astype(str)
    out["method"] = out["method_internal"].map(paper_method_label)
    # Make diagnostic status explicit for Top-K candidate-set rows.
    out["is_deployable_top1"] = ~out["method"].astype(str).str.contains("candidate_set", regex=False)
    out.loc[out["method"].astype(str).str.contains("candidate_set", regex=False), "is_deployable_top1"] = False
    return out
