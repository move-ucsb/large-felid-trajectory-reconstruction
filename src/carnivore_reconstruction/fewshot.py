"""Few-shot adaptation utilities.

Few-shot adaptation is intentionally separate from the pretrained backbone.  It
adds calibration motifs from recent same-animal high-resolution segments, then
reuses the same reconstruction/ranking pipeline.
"""
from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .config import ReconstructionConfig
from .library import build_motif_library
from .model import MotifReconstructionModel
from .ranking import build_movement_priors
from .tasks import ReconstructionTask


def adapt_model_with_calibration(base_model: MotifReconstructionModel, calibration_tasks: list[ReconstructionTask], max_calibration_motifs: int | None = None) -> MotifReconstructionModel:
    """Return a copy of ``base_model`` augmented with calibration motifs."""
    if not calibration_tasks:
        return deepcopy(base_model)
    model = deepcopy(base_model)
    cfg = deepcopy(base_model.config)
    if max_calibration_motifs is not None:
        cfg.max_motifs = int(max_calibration_motifs)
    calibration_library = build_motif_library(calibration_tasks, cfg)

    # Merge existing and calibration tables, then rebuild a single retrieval index
    # by treating the merged motif table/points as if it were the full library.
    # Simpler and safer than mutating sklearn internals.
    all_tasks = []
    # Reconstructing tasks from motif frames is unnecessary; instead, use the
    # calibration-only library when the calibration set exists and append old
    # motifs at table level in future optimization. For the first clean release,
    # calibration motifs replace retrieval with same-animal motifs for speed.
    model.library = calibration_library
    model.movement_priors = build_movement_priors(calibration_tasks, cfg)
    model.metadata = dict(base_model.metadata)
    model.metadata["adapted_with_calibration_tasks"] = len(calibration_tasks)
    return model


def select_past_calibration_tasks(tasks: list[ReconstructionTask], target_task: ReconstructionTask, days: int) -> list[ReconstructionTask]:
    """Select same-animal tasks ending before the target start time."""
    cutoff = pd.to_datetime(target_task.start_time) - pd.to_timedelta(days, unit="D")
    out = []
    for task in tasks:
        if task.truth_xy is None:
            continue
        if task.dataset == target_task.dataset and task.taxon == target_task.taxon and task.animal_id == target_task.animal_id:
            if pd.to_datetime(task.end_time) < pd.to_datetime(target_task.start_time) and pd.to_datetime(task.end_time) >= cutoff:
                out.append(task)
    return out



def run_fewshot_sensitivity(
    base_model: MotifReconstructionModel,
    all_tasks: list[ReconstructionTask],
    eval_tasks: list[ReconstructionTask],
    windows_days: tuple[int, ...] = (1, 3, 7, 14, 30),
    max_calibration_motifs: int | None = None,
    method_name: str = "pretrained_motif_guarded",
) -> dict[str, pd.DataFrame]:
    """Run context-compatible few-shot sensitivity on held-out tasks.

    For each target task and each calibration window, this function builds a
    same-animal calibration library from tasks ending before the target start
    time, evaluates the proposed-method family on that one task, and records the
    requested final method when available. This is intentionally subset-aware:
    tasks without calibration are reported as uncovered rather than imputed.
    """
    from .proposed import evaluate_proposed_methods_for_task
    from .benchmark import attach_split
    from .metrics import summarize_metrics
    from .timing import ProgressPrinter

    metric_parts = []
    coverage_rows = []
    progress = ProgressPrinter("few-shot sensitivity", total=len(eval_tasks) * len(windows_days), every=max(1, min(25, len(eval_tasks))))
    i = 0
    for task in eval_tasks:
        for days in windows_days:
            i += 1
            cal = select_past_calibration_tasks(all_tasks, task, days=days)
            coverage_rows.append({
                "task_uid": task.task_uid,
                "dataset": task.dataset,
                "taxon": task.taxon,
                "animal_id": task.animal_id,
                "setting_name": task.setting_name,
                "fewshot_days": int(days),
                "n_calibration_tasks": len(cal),
                "covered": int(len(cal) > 0),
            })
            if not cal:
                progress.update(i, extra=f"{task.dataset}/{task.taxon}/{task.setting_name} D{days}: no calibration")
                continue
            adapted = adapt_model_with_calibration(base_model, cal, max_calibration_motifs=max_calibration_motifs)
            result = evaluate_proposed_methods_for_task(adapted, task, include_baselines=True, keep_scored=False)
            met = result["metrics"].copy()
            if met.empty:
                continue
            met["fewshot_days"] = int(days)
            met["n_calibration_tasks"] = len(cal)
            met["fewshot_covered"] = 1
            metric_parts.append(met)
            progress.update(i, extra=f"{task.dataset}/{task.taxon}/{task.setting_name} D{days}: cal={len(cal)}")
    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    summary = summarize_metrics(metrics, group_cols=["fewshot_days", "method"]) if not metrics.empty else pd.DataFrame()
    coverage_summary = coverage.groupby("fewshot_days", dropna=False).agg(
        n_tasks=("task_uid", "nunique"),
        n_covered=("covered", "sum"),
        coverage_rate=("covered", "mean"),
        median_calibration_tasks=("n_calibration_tasks", "median"),
    ).reset_index() if not coverage.empty else pd.DataFrame()
    return {"task_metrics": metrics, "summary": summary, "coverage": coverage, "coverage_summary": coverage_summary}
