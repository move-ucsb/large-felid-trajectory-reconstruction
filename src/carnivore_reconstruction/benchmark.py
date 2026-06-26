"""Benchmark helpers for paper-facing accuracy notebooks."""
from __future__ import annotations

import pandas as pd

from .metrics import summarize_metrics, add_linear_baseline_metrics
from .model import MotifReconstructionModel
from .tasks import ReconstructionTask
from .timing import status
from .v31_direct_generators import make_v31_direct_topk_diagnostics, make_v32_expanded_candidate_set_diagnostics
from .ecological_candidates import make_balanced_pooled_top10_diagnostics
from .proposed import evaluate_proposed_methods_for_tasks, summarize_methods_for_selection, add_guarded_methods_from_validation, make_robust_global_selection, make_adaptive_task_selection, make_oracle_distilled_selection, make_conservative_oracle_gap_selection, make_v27c_diverse_topk_diagnostics


def attach_split(metrics: pd.DataFrame, task_table: pd.DataFrame | None) -> pd.DataFrame:
    """Attach split metadata to a metric table without creating suffix columns."""
    if metrics.empty or task_table is None or task_table.empty:
        return metrics
    base_meta_cols = [
        "dataset", "taxon", "animal_id", "setting_name", "split",
        "habitat_id", "study_system", "species_common_name", "species_id",
        "species_group", "genus_group", "transfer_unit", "sex", "age_class",
        "metadata_source",
    ]
    transfer_meta_cols = [
        c for c in task_table.columns
        if c.startswith("best_train_transfer_relation")
        or c.startswith("n_train_")
        or c.startswith("has_train_")
    ]
    meta_cols = base_meta_cols + [c for c in transfer_meta_cols if c not in base_meta_cols]
    base = metrics.drop(columns=[c for c in meta_cols if c in metrics.columns], errors="ignore")
    meta = task_table[["task_uid"] + [c for c in meta_cols if c in task_table.columns]].drop_duplicates("task_uid")
    return base.merge(meta, on="task_uid", how="left")


def run_accuracy_benchmark(
    model: MotifReconstructionModel,
    tasks: list[ReconstructionTask],
    keep_scored: bool = False,
    task_table: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate the publication-focused reconstruction methods.

    The full candidate-bank rows are still saved in the audit metric table, but
    the method-construction logic is intentionally compact:

    - external baselines;
    - robust validation-selected deployable Top-1;
    - compact Top-K diagnostic rows useful for comparison;
    - balanced pooled Top-10 as the paper-facing probabilistic candidate set.

    Earlier experimental selectors that were not retained for publication
    (oracle-distilled, adaptive task switching, conservative oracle-gap switching,
    and setting-specific guarded variants) are intentionally omitted here.
    """
    status(f"Running publication accuracy benchmark on {len(tasks):,} task(s)")
    out = evaluate_proposed_methods_for_tasks(model, tasks, include_baselines=True, keep_scored=keep_scored)
    metrics = attach_split(out["task_metrics"], task_table)
    metrics = add_linear_baseline_metrics(metrics)

    choices = pd.DataFrame()

    # Final deployable Top-1: validation-selected robust global method.
    robust_global, robust_choice = make_robust_global_selection(metrics)
    if robust_global is not None and not robust_global.empty:
        metrics = pd.concat([metrics, robust_global], ignore_index=True, sort=False)
        metrics = add_linear_baseline_metrics(metrics)
    if robust_choice is not None and not robust_choice.empty:
        choices = pd.concat([choices, robust_choice], ignore_index=True, sort=False)

    # Useful compact candidate-set diagnostics for audit/comparison.
    for label, maker in [
        ("diverse Top-K", make_v27c_diverse_topk_diagnostics),
        ("latent direct Top-K", make_v31_direct_topk_diagnostics),
        ("expanded SR Top-K", make_v32_expanded_candidate_set_diagnostics),
        ("balanced pooled Top-10", make_balanced_pooled_top10_diagnostics),
    ]:
        try:
            rows, choice = maker(metrics)
        except Exception as exc:
            status(f"{label} diagnostics skipped after error: {exc}")
            rows, choice = pd.DataFrame(), pd.DataFrame()
        if rows is not None and not rows.empty:
            metrics = pd.concat([metrics, rows], ignore_index=True, sort=False)
            metrics = add_linear_baseline_metrics(metrics)
        if choice is not None and not choice.empty:
            choices = pd.concat([choices, choice], ignore_index=True, sort=False)

    out["task_metrics"] = metrics
    out["summary"] = summarize_metrics(metrics)
    out["setting_summary"] = summarize_metrics(metrics, group_cols=["dataset", "taxon", "setting_name", "method"])
    out["proposed_selection_summary"] = summarize_methods_for_selection(metrics, ["split", "method"]) if "split" in metrics.columns else pd.DataFrame()
    out["setting_choices"] = choices
    return out
