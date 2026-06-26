"""Deployable candidate scoring and guarded path selection."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ReconstructionConfig
from .geometry import directness, path_length, path_ratio, path_step_stats, signed_lateral_stats
from .tasks import ReconstructionTask
from .context import preferred_lateral_sign_from_task


# The scoring profile is deliberately deployable: it uses only endpoint context,
# candidate geometry, and priors estimated from the training split.  A key This release
# change is that detour is no longer penalized toward a straight line.  Instead,
# candidates are compared with a training-derived path-ratio/directness prior for
# the corresponding dataset/taxon/setting when available.  This prevents the
# proposed method from collapsing to linear paths on coarse gaps.
DEFAULT_PROFILE = {
    # The profile is tuned for timestamp-wise accuracy and efficient paths.
    # Detours are not rewarded generically; they are supported only when
    # context-matched training motifs and local heading context agree.
    "motif": 1.00,
    "step": 0.95,
    "turn": 0.35,
    "detour": 0.45,
    "directness": 0.20,
    "lateral": 0.25,
    "timegeo": 1.10,
    "progress": 0.35,
    "origin": 0.45,
    "context": 0.60,
    "direction": 0.25,
    "efficiency": 0.35,
    "source_shape": 0.25,
}


def build_movement_priors(tasks: list[ReconstructionTask], config: ReconstructionConfig) -> pd.DataFrame:
    """Estimate movement priors by dataset/taxon/setting and fallback levels.

    Besides step-length quantiles, This release stores endpoint-conditioned path-shape
    priors (path-ratio and directness quantiles).  These priors are learned only
    from training truth paths and are used to score whether a candidate is too
    linear or too tortuous for the current sampling setting.
    """
    step_rows = []
    task_rows = []
    for task in tasks:
        if task.truth_xy is None or len(task.truth_xy) < 2:
            continue
        disp = float(np.linalg.norm(task.end_xy - task.start_xy))
        truth_len = path_length(task.truth_xy)
        truth_ratio = truth_len / max(disp, 1e-9)
        truth_directness = directness(task.truth_xy)
        task_rows.append({
            "dataset": task.dataset,
            "taxon": task.taxon,
            "setting_name": task.setting_name,
            "path_ratio": float(truth_ratio) if np.isfinite(truth_ratio) else np.nan,
            "directness": float(truth_directness) if np.isfinite(truth_directness) else np.nan,
            "truth_path_length_m": float(truth_len),
            "displacement_m": float(disp),
        })
        steps = np.linalg.norm(np.diff(task.truth_xy, axis=0), axis=1)
        for step in steps:
            step_rows.append({
                "dataset": task.dataset,
                "taxon": task.taxon,
                "setting_name": task.setting_name,
                "step_m": float(step),
            })
    step_df = pd.DataFrame(step_rows)
    task_df = pd.DataFrame(task_rows)
    if step_df.empty:
        return pd.DataFrame([{
            "level": "global", "key": "global", "step_q50": 1.0, "step_q90": 1.0, "step_q95": 1.0,
            "path_ratio_q25": 1.0, "path_ratio_q50": 1.0, "path_ratio_q75": 1.0, "path_ratio_q90": 1.0,
            "directness_q25": 1.0, "directness_q50": 1.0, "directness_q75": 1.0,
        }])

    out = []
    group_specs = [
        ("dataset_taxon_setting", ["dataset", "taxon", "setting_name"]),
        ("dataset_taxon", ["dataset", "taxon"]),
        ("taxon", ["taxon"]),
        ("global", []),
    ]
    for level, cols in group_specs:
        step_groups = [("global", step_df)] if not cols else step_df.groupby(cols, dropna=False)
        task_groups = {"global": task_df} if not cols else {k: g for k, g in task_df.groupby(cols, dropna=False)}
        for key, g in step_groups:
            key_tuple = key if isinstance(key, tuple) else (key,)
            key_str = "|".join(map(str, key_tuple))
            vals = g["step_m"].dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            if len(vals) == 0:
                continue
            tg = task_groups.get(key, pd.DataFrame()) if cols else task_groups.get("global", pd.DataFrame())
            ratios = pd.to_numeric(tg.get("path_ratio", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
            dirs = pd.to_numeric(tg.get("directness", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
            dirs = dirs[np.isfinite(dirs)]
            row = {
                "level": level,
                "key": key_str,
                "n_steps": int(len(vals)),
                "n_tasks": int(len(tg)) if tg is not None else 0,
                "step_q50": float(np.nanquantile(vals, 0.50)),
                "step_q90": float(np.nanquantile(vals, config.step_capacity_quantile)),
                "step_q95": float(np.nanquantile(vals, 0.95)),
                "path_ratio_q25": float(np.nanquantile(ratios, 0.25)) if len(ratios) else 1.0,
                "path_ratio_q50": float(np.nanquantile(ratios, 0.50)) if len(ratios) else 1.0,
                "path_ratio_q75": float(np.nanquantile(ratios, 0.75)) if len(ratios) else 1.0,
                "path_ratio_q90": float(np.nanquantile(ratios, 0.90)) if len(ratios) else 1.0,
                "directness_q25": float(np.nanquantile(dirs, 0.25)) if len(dirs) else 1.0,
                "directness_q50": float(np.nanquantile(dirs, 0.50)) if len(dirs) else 1.0,
                "directness_q75": float(np.nanquantile(dirs, 0.75)) if len(dirs) else 1.0,
            }
            if cols:
                for c, v in zip(cols, key_tuple):
                    row[c] = v
            out.append(row)
    return pd.DataFrame(out)


def _prior_row_for_task(task: ReconstructionTask, priors: pd.DataFrame) -> pd.Series | None:
    candidates = [
        ("dataset_taxon_setting", f"{task.dataset}|{task.taxon}|{task.setting_name}"),
        ("dataset_taxon", f"{task.dataset}|{task.taxon}"),
        ("taxon", f"{task.taxon}"),
        ("global", "global"),
    ]
    if priors is None or priors.empty:
        return None
    for level, key in candidates:
        sub = priors[priors["level"].eq(level) & priors["key"].astype(str).eq(str(key))]
        if not sub.empty:
            return sub.iloc[0]
    return None


def step_capacity_for_task(task: ReconstructionTask, priors: pd.DataFrame, config: ReconstructionConfig) -> float:
    """Find a robust step-capacity prior for a task."""
    row = _prior_row_for_task(task, priors)
    if row is not None and "step_q90" in row.index:
        cap = float(row.get("step_q90", 1.0)) * config.step_capacity_min_multiplier
        return max(cap, 1.0)
    return 1.0


def shape_priors_for_task(task: ReconstructionTask, priors: pd.DataFrame) -> dict:
    """Return training-derived expected path ratio/directness for scoring."""
    row = _prior_row_for_task(task, priors)
    if row is None:
        return {"target_path_ratio": 1.0, "target_directness": 1.0, "prior_level": "none"}
    ratio = float(row.get("path_ratio_q50", 1.0)) if pd.notna(row.get("path_ratio_q50", np.nan)) else 1.0
    direct = float(row.get("directness_q50", 1.0)) if pd.notna(row.get("directness_q50", np.nan)) else 1.0
    # Keep targets in safe ranges.  The lower bound prevents zero-displacement
    # artifacts from producing an impossible prior; the upper bound prevents a
    # few highly tortuous tracks from making the model choose wild detours.
    ratio = float(np.clip(ratio, 1.0, 6.0))
    direct = float(np.clip(direct, 0.02, 1.0))
    return {
        "target_path_ratio": ratio,
        "target_directness": direct,
        "prior_level": str(row.get("level", "unknown")),
        "target_path_ratio_q25": float(row.get("path_ratio_q25", np.nan)),
        "target_path_ratio_q75": float(row.get("path_ratio_q75", np.nan)),
    }


def _finite_float(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def candidate_feature_row(task: ReconstructionTask, path: np.ndarray, candidate_row: pd.Series | dict, step_capacity: float, shape_prior: dict | None = None) -> dict:
    """Proposed-method deployable cost components for one proposed candidate.

    V20 focuses on timestamp-wise reconstruction usefulness.  The target path
    ratio remains efficient by default; source-motif detours affect the expected
    shape only when they come from context-matched retrieved motifs.  This keeps
    generated paths close to the hidden trajectory in time-indexed space while
    avoiding gratuitous detours.
    """
    shape_prior = shape_prior or {}
    target_ratio_prior = float(shape_prior.get("target_path_ratio", 1.0))
    target_direct = float(shape_prior.get("target_directness", 1.0))
    length = path_length(path)
    disp = float(np.linalg.norm(task.end_xy - task.start_xy))
    ratio = length / max(disp, 1e-9)
    cand_direct = directness(path)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.array([0.0])
    violation = np.maximum(steps - step_capacity, 0.0)
    lateral = signed_lateral_stats(path, task.start_xy, task.end_xy)
    step_stats = path_step_stats(path)

    expected_step = max(disp / max(len(path) - 1, 1), 1.0)
    expected_step = min(max(expected_step, 1.0), max(step_capacity, 1.0))
    step_mean = float(step_stats["step_mean"])
    step_q90 = float(step_stats["step_q90"])
    step_cost = abs(np.log((step_mean + 1e-6) / (expected_step + 1e-6))) + max(0.0, step_q90 / max(step_capacity, 1e-6) - 1.0)

    turn_cost = 0.0
    if len(path) > 2:
        v1 = np.diff(path[:-1], axis=0)
        v2 = np.diff(path[1:], axis=0)
        denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        ok = denom > 1e-9
        if ok.any():
            ang = np.full(len(denom), np.nan)
            ang[ok] = np.arccos(np.clip(np.sum(v1[ok] * v2[ok], axis=1) / denom[ok], -1, 1))
            turn_cost = float(np.nanmean(np.abs(ang)))

    style = str(candidate_row.get("style", "")) if hasattr(candidate_row, "get") else ""
    source_ratio = _finite_float(candidate_row.get("source_path_ratio", np.nan) if hasattr(candidate_row, "get") else np.nan)
    source_direct = _finite_float(candidate_row.get("source_directness", np.nan) if hasattr(candidate_row, "get") else np.nan)
    source_lat_sign = _finite_float(candidate_row.get("source_lateral_sign", 0.0) if hasattr(candidate_row, "get") else 0.0, 0.0)
    source_lat_abs = _finite_float(candidate_row.get("source_lateral_abs_mean_norm", 0.0) if hasattr(candidate_row, "get") else 0.0, 0.0)
    is_retrieved = style == "retrieved_motif"

    # Context mismatch already enters retrieval_score, but we keep it as an
    # explicit scoring term so selected-candidate diagnostics show why a source
    # motif was preferred.
    ctx_temporal = _finite_float(candidate_row.get("context_temporal_cost", 0.0) if hasattr(candidate_row, "get") else 0.0, 0.0)
    ctx_environment = _finite_float(candidate_row.get("context_environment_cost", 0.0) if hasattr(candidate_row, "get") else 0.0, 0.0)
    ctx_demographic = _finite_float(candidate_row.get("context_demographic_cost", 0.0) if hasattr(candidate_row, "get") else 0.0, 0.0)
    context_cost = (
        ctx_temporal * 0.40
        + ctx_environment * 0.45
        + ctx_demographic * 0.25
    )

    max_pref_ratio = float(getattr(config, "context_max_preferred_path_ratio", 2.25)) if False else 2.25
    # NOTE: this local default is overwritten in score_candidates via profile
    # weights; the preferred ratio itself must stay independent of hidden truth.
    # Expected shape blends an efficient path, training setting prior, and source
    # motif ratio.  Highly tortuous source motifs are clipped to prevent visually
    # impressive but timestamp-wise poor paths.
    if is_retrieved and np.isfinite(source_ratio) and source_ratio > 0:
        expected_ratio = 0.50 * 1.0 + 0.25 * min(max(target_ratio_prior, 1.0), 2.5) + 0.25 * min(max(source_ratio, 1.0), 2.5)
    else:
        expected_ratio = 0.70 * 1.0 + 0.30 * min(max(target_ratio_prior, 1.0), 2.0)
    expected_ratio = float(np.clip(expected_ratio, 1.0, 2.50))

    detour_cost = float(abs(np.log(max(ratio, 1e-9) / max(expected_ratio, 1e-9))))
    directness_cost = float(abs(cand_direct - target_direct)) if np.isfinite(cand_direct) else 0.0
    # Efficiency regularizer: path ratios close to 1 are favored unless context
    # clearly supports detouring. This is important for downstream interaction
    # inference where timestamp-wise positions matter more than scenic realism.
    efficiency_cost = float(abs(np.log(max(ratio, 1e-9))))
    excess_detour_cost = float(max(0.0, np.log(max(ratio, 1e-9) / max(expected_ratio * 1.10, 1.0))))
    detour_from_linear_cost = float(abs(np.log(max(ratio, 1e-9))))

    lateral_cost = lateral["mean_abs_lateral_norm"] + 0.5 * lateral["max_abs_lateral_norm"]
    progress_cost = lateral["backtrack_fraction"]
    timegeo_cost = float(np.nanmean((violation / max(step_capacity, 1.0)) ** 2)) if len(violation) else 0.0

    cand_sign = float(lateral.get("lateral_sign", 0.0))
    pref_sign = preferred_lateral_sign_from_task(task)
    direction_cost = 0.0
    if lateral.get("mean_abs_lateral_norm", 0.0) >= 0.03:
        # Prefer adjacent-heading sign if available; otherwise use source motif
        # sign. If no directional information exists, do not penalize side.
        expected_sign = pref_sign if pref_sign != 0 else (source_lat_sign if source_lat_abs >= 0.02 else 0.0)
        if expected_sign != 0 and cand_sign != 0 and np.sign(cand_sign) != np.sign(expected_sign):
            direction_cost = 1.0

    # Source-shape support: a candidate detour should look like a context-matched
    # source motif. Internal linear/heading candidates get zero source support.
    if is_retrieved and np.isfinite(source_ratio) and source_ratio > 0:
        source_shape_cost = float(abs(np.log(max(ratio, 1e-9) / max(min(source_ratio, 2.5), 1e-9))))
    else:
        source_shape_cost = 0.0

    retrieval_score = float(candidate_row.get("retrieval_score", 0.0)) if hasattr(candidate_row, "get") else 0.0
    origin_cost = 0.0
    if style == "direct_identity":
        retrieval_score = max(retrieval_score, 0.16)
        origin_cost = 0.35  # not too large; direct paths are valid when context is weak
    elif style == "heading_continuity":
        retrieval_score = max(retrieval_score, 0.20)
        origin_cost = 0.25
    elif style == "fallback_linear":
        retrieval_score = max(retrieval_score, 1.0)
        origin_cost = 1.00

    row = {
        "path_length_m": length,
        "path_ratio": ratio,
        "directness": cand_direct,
        "target_path_ratio": target_ratio_prior,
        "expected_path_ratio": expected_ratio,
        "target_directness": target_direct,
        "shape_prior_level": shape_prior.get("prior_level", "none"),
        "source_path_ratio": source_ratio,
        "source_directness": source_direct,
        "source_lateral_sign": source_lat_sign,
        "candidate_lateral_sign": cand_sign,
        "preferred_lateral_sign": pref_sign,
        "step_capacity_m": step_capacity,
        "step_violation_mean_m": float(np.nanmean(violation)) if len(violation) else 0.0,
        "step_violation_fraction": float(np.mean(steps > step_capacity)) if len(steps) else 0.0,
        "step_q90_over_capacity": float(step_q90 / max(step_capacity, 1e-9)),
        "detour_cost": detour_cost,
        "directness_cost": directness_cost,
        "detour_from_linear_cost": detour_from_linear_cost,
        "efficiency_cost": efficiency_cost,
        "excess_detour_cost": excess_detour_cost,
        "direction_cost": direction_cost,
        "context_cost": context_cost,
        "context_temporal_cost": ctx_temporal,
        "context_environment_cost": ctx_environment,
        "context_demographic_cost": ctx_demographic,
        "source_shape_cost": source_shape_cost,
        "lateral_cost": lateral_cost,
        "progress_cost": progress_cost,
        "timegeo_cost": timegeo_cost,
        "step_cost": step_cost,
        "turn_cost": turn_cost,
        "motif_cost": retrieval_score,
        "origin_cost": origin_cost,
        "candidate_cost_motif": retrieval_score,
        "candidate_cost_step": step_cost,
        "candidate_cost_turn": turn_cost,
        "candidate_cost_detour": detour_cost,
        "candidate_cost_directness": directness_cost,
        "candidate_cost_lateral": lateral_cost,
        "candidate_cost_timegeo": timegeo_cost,
        "candidate_cost_progress": progress_cost,
        "candidate_cost_origin": origin_cost,
        "candidate_cost_context": context_cost,
        "candidate_cost_direction": direction_cost,
        "candidate_cost_efficiency": efficiency_cost,
        "candidate_cost_source_shape": source_shape_cost,
        "candidate_step_cap_m": step_capacity,
    }
    return row

def score_candidates(task: ReconstructionTask, candidate_table: pd.DataFrame, candidate_paths: dict[str, np.ndarray], priors: pd.DataFrame, config: ReconstructionConfig, profile: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Score proposed candidates using deployable cost components."""
    t0 = time.perf_counter()
    profile = dict(DEFAULT_PROFILE if profile is None else profile)
    step_capacity = step_capacity_for_task(task, priors, config)
    shape_prior = shape_priors_for_task(task, priors)
    rows = []
    for _, row in candidate_table.iterrows():
        cid = str(row["candidate_id"])
        features = candidate_feature_row(task, candidate_paths[cid], row, step_capacity, shape_prior=shape_prior)
        total_cost = (
            profile.get("motif", 1.0) * features["motif_cost"]
            + profile.get("step", 1.0) * features["step_cost"]
            + profile.get("turn", 0.0) * features["turn_cost"]
            + profile.get("detour", 1.0) * features["detour_cost"]
            + profile.get("directness", 0.0) * features["directness_cost"]
            + profile.get("lateral", 1.0) * features["lateral_cost"]
            + profile.get("timegeo", 1.0) * features["timegeo_cost"]
            + profile.get("progress", 1.0) * features["progress_cost"]
            + profile.get("origin", 0.0) * features["origin_cost"]
            + profile.get("context", 0.0) * features.get("context_cost", 0.0)
            + profile.get("direction", 0.0) * features.get("direction_cost", 0.0)
            + profile.get("efficiency", 0.0) * features.get("efficiency_cost", 0.0)
            + profile.get("source_shape", 0.0) * features.get("source_shape_cost", 0.0)
        )
        out = row.to_dict()
        out.update(features)
        out["proposal_cost"] = float(total_cost)
        out["proposal_score"] = -float(total_cost)
        out["cost_rank_score"] = -float(total_cost)
        out["total_cost"] = float(total_cost)
        rows.append(out)
    scored = pd.DataFrame(rows)
    if scored.empty:
        timing = {"task_uid": task.task_uid, "score_seconds": time.perf_counter() - t0}
        return scored, timing
    scored = scored.sort_values(["proposal_cost", "candidate_order"], kind="mergesort").reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    z = -pd.to_numeric(scored["proposal_cost"], errors="coerce").fillna(1e6).to_numpy(dtype=float)
    z = z - np.nanmax(z)
    w = np.exp(z)
    sw = float(np.nansum(w))
    scored["listwise_probability"] = w / sw if np.isfinite(sw) and sw > 0 else 1.0 / max(len(scored), 1)
    timing = {"task_uid": task.task_uid, "score_seconds": time.perf_counter() - t0}
    return scored, timing


def softmax_weights(costs: np.ndarray, beta: float) -> np.ndarray:
    costs = np.asarray(costs, dtype=float)
    if len(costs) == 0:
        return costs
    z = -costs / max(float(beta), 1e-9)
    z = z - np.nanmax(z)
    w = np.exp(z)
    if not np.all(np.isfinite(w)) or np.nansum(w) <= 0:
        return np.ones(len(costs)) / len(costs)
    return w / np.nansum(w)


def select_paths(scored: pd.DataFrame, candidate_paths: dict[str, np.ndarray], config: ReconstructionConfig) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Apply guards and return Top-K plus a weighted representative path."""
    if scored.empty:
        raise ValueError("Cannot select paths from an empty candidate table.")
    usable = scored.copy()
    guard_mask = (
        usable["path_ratio"].astype(float).le(config.max_path_ratio)
        & usable["step_violation_fraction"].astype(float).le(config.max_step_violation_fraction)
    )
    if guard_mask.any():
        usable = usable[guard_mask].copy()
    else:
        usable = scored.sort_values(["step_violation_fraction", "path_ratio", "total_cost"], kind="mergesort").head(max(config.output_top_k, 1)).copy()

    usable = usable.sort_values(["total_cost", "candidate_order"], kind="mergesort").reset_index(drop=True)
    top = usable.head(config.output_top_k).copy()
    top["selected_rank"] = np.arange(1, len(top) + 1)

    rep_source = usable.head(min(config.representative_top_k, len(usable))).copy()
    weights = softmax_weights(rep_source["total_cost"].to_numpy(dtype=float), config.softmax_beta)
    paths = [candidate_paths[cid] for cid in rep_source["candidate_id"].astype(str)]
    min_len = min(len(p) for p in paths)
    rep = np.zeros((min_len, 2), dtype=float)
    for w, p in zip(weights, paths):
        rep += w * p[:min_len]

    selected_paths = {str(cid): candidate_paths[str(cid)] for cid in top["candidate_id"]}
    selected_paths["representative"] = rep
    summary = pd.DataFrame([{
        "task_uid": str(scored["task_uid"].iloc[0]),
        "n_candidates": int(len(scored)),
        "n_usable_after_guards": int(len(usable)),
        "top_candidate_id": str(top["candidate_id"].iloc[0]) if len(top) else None,
        "top_origin": str(top["origin"].iloc[0]) if len(top) else None,
        "top_total_cost": float(top["total_cost"].iloc[0]) if len(top) else np.nan,
        "representative_source_k": int(len(rep_source)),
    }])
    return top, selected_paths, summary
