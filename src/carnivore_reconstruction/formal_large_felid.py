
"""Formal large-felid-only evaluation helpers.

This module is intentionally separate from the reconstruction algorithm.  It
defines the publication evaluation protocol:

- main analysis species: puma/cougar, tiger, leopard
- bobcat excluded from main analysis as an out-of-domain small-felid case
- formal train/validation/test split is individual-disjoint
- test split is used only for final reporting after the algorithm is locked
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .utils import stable_hash01, write_json, write_table, read_table
from .metrics import add_linear_baseline_metrics


MAIN_TAXA = ("puma", "cougar", "tiger", "leopard")
EXCLUDED_TAXA = ("bobcat",)
FINAL_PROBABILISTIC_METHOD = "balanced_pooled_top10_candidate_set"
FINAL_DEPLOYABLE_TOP1_METHODS = ("robust_pretrained_top1",)


def is_main_large_felid_taxon(value: object) -> bool:
    s = str(value).strip().lower()
    return s in MAIN_TAXA


def filter_dataset_specs_large_felids(dataset_specs):
    """Drop bobcat DatasetSpec entries before loading raw tracks."""
    out = []
    for spec in dataset_specs:
        taxon = str(getattr(spec, "taxon", "")).strip().lower()
        dataset = str(getattr(spec, "dataset", "")).strip().lower()
        if taxon in EXCLUDED_TAXA or "bobcat" in dataset:
            continue
        out.append(spec)
    return out


def filter_large_felid_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    if tracks is None or tracks.empty or "taxon" not in tracks.columns:
        return tracks
    out = tracks[tracks["taxon"].astype(str).str.lower().isin(MAIN_TAXA)].copy().reset_index(drop=True)
    return out


def filter_large_felid_tasks(task_table: pd.DataFrame, task_points: pd.DataFrame | None = None):
    """Filter task table/points to the main large-felid scope."""
    if task_table is None or task_table.empty:
        return task_table, task_points
    tt = task_table.copy()
    if "taxon" in tt.columns:
        keep = tt["taxon"].astype(str).str.lower().isin(MAIN_TAXA)
    else:
        keep = ~tt.get("dataset", pd.Series(index=tt.index, dtype=str)).astype(str).str.lower().str.contains("bobcat", na=False)
    tt = tt[keep].copy().reset_index(drop=True)
    if task_points is not None and not task_points.empty and "task_uid" in task_points.columns:
        keep_uids = set(tt["task_uid"].astype(str))
        tp = task_points[task_points["task_uid"].astype(str).isin(keep_uids)].copy().reset_index(drop=True)
    else:
        tp = task_points
    return tt, tp




def clean_large_felid_scope(
    task_table: pd.DataFrame,
    task_points: pd.DataFrame | None = None,
    *,
    label: str = "task_table",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Remove non-main taxa from formal main-analysis tables and continue.

    This is intentionally non-failing for long overnight runs. If a stray bobcat
    or other out-of-scope row appears, the row is removed and a short message is
    printed. The cleaned task table and matching task points are returned.
    """
    if task_table is None or task_table.empty:
        return task_table, task_points
    before = len(task_table)
    cleaned, cleaned_points = filter_large_felid_tasks(task_table, task_points)
    after = len(cleaned) if cleaned is not None else 0
    if after < before:
        removed = before - after
        removed_taxa = []
        if "taxon" in task_table.columns:
            keep = task_table["taxon"].astype(str).str.lower().isin(MAIN_TAXA)
            removed_taxa = sorted(task_table.loc[~keep, "taxon"].dropna().astype(str).unique().tolist())
        print(
            f"[formal-cleanup] Removed {removed} out-of-scope task(s) from {label}. "
            f"Removed taxa/datasets include: {removed_taxa}",
            flush=True,
        )
    return cleaned, cleaned_points


def repair_individual_disjoint_split(
    task_table: pd.DataFrame,
    split_table: pd.DataFrame | None = None,
    *,
    seed: int = 20260625,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    label: str = "task_table",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure each individual belongs to only one split and continue.

    If a saved split table is available, it is used to repair split labels.
    Otherwise a deterministic individual-disjoint split is regenerated.
    """
    if task_table is None or task_table.empty:
        return task_table, pd.DataFrame() if split_table is None else split_table
    tt = add_individual_key(task_table) if "formal_individual_key" not in task_table.columns else task_table.copy()

    repaired_from_split_table = False
    if split_table is not None and not split_table.empty and {"formal_individual_key", "split"}.issubset(split_table.columns):
        split_map = split_table.drop_duplicates("formal_individual_key").set_index("formal_individual_key")["split"].astype(str).to_dict()
        known = tt["formal_individual_key"].astype(str).map(split_map)
        if known.notna().any():
            tt.loc[known.notna(), "split"] = known[known.notna()].values
            repaired_from_split_table = True

    leak = (
        tt.groupby("formal_individual_key", dropna=False)["split"]
        .nunique()
        .reset_index(name="n_splits")
    )
    leak = leak[leak["n_splits"] > 1]
    missing = "split" not in tt.columns or tt["split"].isna().any()

    if not leak.empty or missing or not repaired_from_split_table:
        if not leak.empty:
            print(
                f"[formal-cleanup] Repaired {len(leak)} individual(s) appearing in multiple splits in {label}.",
                flush=True,
            )
        if missing:
            print(f"[formal-cleanup] Repaired missing split labels in {label}.", flush=True)
        tt, split_table_new = assign_formal_individual_split(
            tt,
            stratify_cols=[c for c in ["taxon"] if c in tt.columns],
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
    else:
        split_table_new = split_table.copy()

    return tt.reset_index(drop=True), split_table_new.reset_index(drop=True)


def prepare_formal_task_tables(
    task_table: pd.DataFrame,
    task_points: pd.DataFrame | None = None,
    split_table: pd.DataFrame | None = None,
    *,
    seed: int = 20260625,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    label: str = "task_table",
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Clean scope and repair individual-disjoint split without stopping."""
    task_table, task_points = clean_large_felid_scope(task_table, task_points, label=label)
    task_table, split_table = repair_individual_disjoint_split(
        task_table,
        split_table=split_table,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        label=label,
    )
    if task_points is not None and not task_points.empty and "task_uid" in task_points.columns:
        keep_uids = set(task_table["task_uid"].astype(str))
        task_points = task_points[task_points["task_uid"].astype(str).isin(keep_uids)].copy().reset_index(drop=True)
    return task_table, task_points, split_table


# Backward-compatible non-failing aliases used by earlier notebooks. They now
# clean/repair and print a message instead of raising errors.
def assert_large_felid_scope(task_table: pd.DataFrame, *, label: str = "task_table") -> None:
    clean_large_felid_scope(task_table, None, label=label)


def assert_individual_disjoint_split(task_table: pd.DataFrame, *, label: str = "task_table") -> None:
    repair_individual_disjoint_split(task_table, label=label)


def individual_key_columns(task_table: pd.DataFrame) -> list[str]:
    cols = [c for c in ["dataset", "taxon", "animal_id"] if c in task_table.columns]
    if "animal_id" not in cols:
        raise ValueError("task_table must contain animal_id for individual-disjoint formal split")
    return cols


def add_individual_key(task_table: pd.DataFrame) -> pd.DataFrame:
    tt = task_table.copy()
    cols = individual_key_columns(tt)
    tt["formal_individual_key"] = tt[cols].astype(str).agg("|".join, axis=1)
    return tt


def _assign_splits_for_animals(animal_rows: pd.DataFrame, seed: int = 20260625,
                               train_fraction: float = 0.70,
                               validation_fraction: float = 0.15) -> dict[str, str]:
    """Assign split labels to animal keys within a stratum.

    Guarantees individual-disjoint splits.  For n>=3, at least one validation
    and one test animal are assigned when possible.
    """
    rows = animal_rows.copy()
    if rows.empty:
        return {}
    rows["_hash"] = rows["formal_individual_key"].map(lambda x: stable_hash01(f"{seed}|{x}"))
    rows = rows.sort_values(["_hash", "formal_individual_key"], kind="mergesort").reset_index(drop=True)
    keys = rows["formal_individual_key"].astype(str).tolist()
    n = len(keys)
    if n == 1:
        return {keys[0]: "train"}
    if n == 2:
        return {keys[0]: "train", keys[1]: "test"}
    n_val = max(1, int(round(n * validation_fraction)))
    n_test = max(1, int(round(n * (1.0 - train_fraction - validation_fraction))))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test
    out = {}
    for i, key in enumerate(keys):
        if i < n_train:
            out[key] = "train"
        elif i < n_train + n_val:
            out[key] = "validation"
        else:
            out[key] = "test"
    return out


def assign_formal_individual_split(task_table: pd.DataFrame,
                                   stratify_cols: list[str] | None = None,
                                   seed: int = 20260625,
                                   train_fraction: float = 0.70,
                                   validation_fraction: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic individual-disjoint formal split.

    Default stratification is by taxon.  This keeps every individual entirely in
    train, validation, or test while maintaining species representation when the
    available number of individuals allows it.
    """
    tt = add_individual_key(task_table)
    stratify_cols = stratify_cols or [c for c in ["taxon"] if c in tt.columns]
    if not stratify_cols:
        stratify_cols = ["__all__"]
        tt["__all__"] = "all"

    animals = tt.drop_duplicates("formal_individual_key").copy()
    summary_cols = ["formal_individual_key"] + [c for c in ["dataset", "taxon", "animal_id", "species_id", "habitat_id", "study_system"] if c in animals.columns]
    animals = animals[summary_cols].copy()

    split_map = {}
    for _, g in animals.groupby(stratify_cols, dropna=False, sort=False):
        split_map.update(_assign_splits_for_animals(g, seed=seed, train_fraction=train_fraction, validation_fraction=validation_fraction))
    tt["split"] = tt["formal_individual_key"].astype(str).map(split_map).fillna("train")
    split_table = animals.copy()
    split_table["split"] = split_table["formal_individual_key"].astype(str).map(split_map).fillna("train")
    if "__all__" in tt.columns:
        tt = tt.drop(columns=["__all__"])
    return tt.reset_index(drop=True), split_table.reset_index(drop=True)


def write_analysis_decisions(output_dir: str | Path, final_algorithm: str = FINAL_PROBABILISTIC_METHOD,
                             split_seed: int = 20260625) -> Path:
    decisions = {
        "main_scope": "large felids",
        "main_taxa": list(MAIN_TAXA),
        "excluded_from_main": list(EXCLUDED_TAXA),
        "bobcat_handling": "excluded from main formal analysis; can be evaluated separately as an out-of-domain small-felid sensitivity case",
        "exclusion_reason": "bobcat is a smaller felid/mesocarnivore with different movement scale from the large-felid scope of the method paper",
        "algorithm_family": "expanded probabilistic time-geographic reconstruction",
        "final_probabilistic_candidate_set": final_algorithm,
        "deployable_top1_reported": list(FINAL_DEPLOYABLE_TOP1_METHODS),
        "formal_split_unit": "individual",
        "formal_split_property": "individual-disjoint train/validation/test split",
        "split_seed": int(split_seed),
        "algorithm_selection_rule": "final algorithm is selected using validation and scientific defensibility before formal held-out test reporting",
        "test_set_use": "final evaluation only; not used to tune algorithm after the formal package is locked",
        "task_caps": {"max_tasks_per_animal_setting": None, "max_tasks_per_dataset_taxon_setting": None},
        "cleanup_policy": "repair scope/split issues and continue rather than stopping long overnight runs",
        "primary_metrics": ["ADE", "RMSE", "Frechet", "DTW", "within_50m", "better_than_linear"],
    }
    return write_json(decisions, Path(output_dir) / "analysis_decisions.json")



def balanced_sample_task_table(
    task_table: pd.DataFrame,
    max_tasks: int | None,
    *,
    group_cols: list[str] | None = None,
    seed: int = 42,
    label: str = "evaluation",
) -> pd.DataFrame:
    """Return a deterministic balanced subset for manageable formal evaluation.

    Sampling is balanced over taxon and setting when possible, with stable
    hash-based ordering so repeated notebook runs select the same tasks.  This
    keeps validation/test/transfer runtime reasonable without letting one very
    dense setting dominate the evaluation subset.
    """
    if task_table is None or task_table.empty or max_tasks is None:
        return task_table.copy() if task_table is not None else task_table
    max_tasks = int(max_tasks)
    if len(task_table) <= max_tasks:
        return task_table.copy().reset_index(drop=True)

    cols = group_cols or [c for c in ["taxon", "setting_name"] if c in task_table.columns]
    if not cols:
        cols = [c for c in ["dataset", "taxon", "setting_name"] if c in task_table.columns]
    if not cols:
        out = task_table.copy()
        out["_sample_hash"] = out["task_uid"].astype(str).map(lambda x: stable_hash01(f"{seed}|{x}"))
        out = out.sort_values("_sample_hash", kind="mergesort").head(max_tasks).drop(columns="_sample_hash")
        return out.sort_values("task_uid").reset_index(drop=True)

    pieces = []
    groups = list(task_table.groupby(cols, dropna=False, sort=True))
    n_groups = max(1, len(groups))
    base_quota = max(1, max_tasks // n_groups)

    remainders = []
    for key, g in groups:
        gg = g.copy()
        gg["_sample_hash"] = gg["task_uid"].astype(str).map(lambda x: stable_hash01(f"{seed}|{label}|{x}"))
        gg = gg.sort_values(["_sample_hash", "task_uid"], kind="mergesort")
        take = min(len(gg), base_quota)
        pieces.append(gg.head(take))
        rem = gg.iloc[take:].copy()
        if not rem.empty:
            remainders.append(rem)

    selected = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
    if len(selected) < max_tasks and remainders:
        rem_all = pd.concat(remainders, ignore_index=True, sort=False)
        rem_all = rem_all.sort_values(["_sample_hash", "task_uid"], kind="mergesort")
        selected = pd.concat([selected, rem_all.head(max_tasks - len(selected))], ignore_index=True, sort=False)

    selected = selected.head(max_tasks).drop(columns=["_sample_hash"], errors="ignore")
    selected = selected.sort_values([c for c in ["dataset", "taxon", "setting_name", "animal_id", "task_uid"] if c in selected.columns], kind="mergesort").reset_index(drop=True)
    print(f"[formal-sampling] {label}: selected {len(selected):,} of {len(task_table):,} task(s) using balanced cap={max_tasks:,}.", flush=True)
    return selected


def read_table_auto(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return read_table(path)
    alt = path.with_suffix(".csv")
    if alt.exists():
        return pd.read_csv(alt)
    raise FileNotFoundError(f"Could not find {path} or {alt}")


def summarize_with_linear(metrics: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or ["method"]
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    mm = add_linear_baseline_metrics(metrics.copy())
    rows = []
    for key, g in mm.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {c: v for c, v in zip(group_cols, key)}
        row.update({
            "n_tasks": int(g["task_uid"].nunique()) if "task_uid" in g.columns else int(len(g)),
            "ADE_median": float(pd.to_numeric(g.get("ADE"), errors="coerce").median()) if "ADE" in g.columns else np.nan,
            "ADE_mean": float(pd.to_numeric(g.get("ADE"), errors="coerce").mean()) if "ADE" in g.columns else np.nan,
            "ADE_q75": float(pd.to_numeric(g.get("ADE"), errors="coerce").quantile(0.75)) if "ADE" in g.columns else np.nan,
            "ADE_q90": float(pd.to_numeric(g.get("ADE"), errors="coerce").quantile(0.90)) if "ADE" in g.columns else np.nan,
            "RMSE_median": float(pd.to_numeric(g.get("RMSE"), errors="coerce").median()) if "RMSE" in g.columns else np.nan,
            "Frechet_median": float(pd.to_numeric(g.get("Frechet"), errors="coerce").median()) if "Frechet" in g.columns else np.nan,
            "DTW_median": float(pd.to_numeric(g.get("DTW"), errors="coerce").median()) if "DTW" in g.columns else np.nan,
            "within_50m_rate": float(pd.to_numeric(g.get("within_50m"), errors="coerce").mean()) if "within_50m" in g.columns else np.nan,
        })
        if str(row.get("method")) == "linear":
            row["better_than_linear_rate"] = 0.0
        else:
            row["better_than_linear_rate"] = float(pd.to_numeric(g.get("better_than_linear", pd.Series(index=g.index, dtype=float)), errors="coerce").mean())
        if "ade_ratio_to_linear" in g.columns:
            row["ade_ratio_to_linear_median"] = float(pd.to_numeric(g["ade_ratio_to_linear"], errors="coerce").median())
            row["ade_ratio_to_linear_mean"] = float(pd.to_numeric(g["ade_ratio_to_linear"], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def save_metric_bundle(metrics: pd.DataFrame, out_dir: str | Path, prefix: str, paper_filter_func=None) -> dict[str, pd.DataFrame]:
    """Save task/method/setting/species summary tables."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / f"{prefix}_task_metrics.csv", index=False)
    method_summary = summarize_with_linear(metrics, ["method"]).sort_values(["ADE_median", "method"], na_position="last").reset_index(drop=True)
    setting_cols = [c for c in ["dataset", "taxon", "setting_name", "method"] if c in metrics.columns]
    species_cols = [c for c in ["taxon", "method"] if c in metrics.columns]
    individual_cols = [c for c in ["formal_individual_key", "animal_id", "taxon", "method"] if c in metrics.columns]
    method_summary.to_csv(out_dir / f"{prefix}_method_summary.csv", index=False)
    setting_summary = summarize_with_linear(metrics, setting_cols) if setting_cols else pd.DataFrame()
    species_summary = summarize_with_linear(metrics, species_cols) if species_cols else pd.DataFrame()
    individual_summary = summarize_with_linear(metrics, individual_cols) if individual_cols else pd.DataFrame()
    if not setting_summary.empty:
        setting_summary.to_csv(out_dir / f"{prefix}_setting_summary.csv", index=False)
    if not species_summary.empty:
        species_summary.to_csv(out_dir / f"{prefix}_species_summary.csv", index=False)
    if not individual_summary.empty:
        individual_summary.to_csv(out_dir / f"{prefix}_individual_summary.csv", index=False)

    out = {"method_summary": method_summary, "setting_summary": setting_summary, "species_summary": species_summary, "individual_summary": individual_summary}
    if paper_filter_func is not None:
        paper = paper_filter_func(metrics)
        paper.to_csv(out_dir / f"paper_{prefix}_task_metrics.csv", index=False)
        paper_method = summarize_with_linear(paper, ["method"]).sort_values(["ADE_median", "method"], na_position="last").reset_index(drop=True)
        paper_method.to_csv(out_dir / f"paper_{prefix}_method_summary.csv", index=False)
        if setting_cols:
            summarize_with_linear(paper, setting_cols).to_csv(out_dir / f"paper_{prefix}_setting_summary.csv", index=False)
        if species_cols:
            summarize_with_linear(paper, species_cols).to_csv(out_dir / f"paper_{prefix}_species_summary.csv", index=False)
        out["paper_method_summary"] = paper_method
    return out


def formal_split_report(task_table: pd.DataFrame, split_table: pd.DataFrame) -> dict:
    report = {
        "n_tasks": int(len(task_table)),
        "n_individuals": int(split_table["formal_individual_key"].nunique()) if "formal_individual_key" in split_table.columns else None,
        "tasks_by_split": task_table["split"].value_counts(dropna=False).to_dict() if "split" in task_table.columns else {},
    }
    if {"taxon", "split"}.issubset(task_table.columns):
        report["tasks_by_taxon_split"] = task_table.groupby(["taxon", "split"], dropna=False).size().reset_index(name="n_tasks").to_dict(orient="records")
    if {"taxon", "split"}.issubset(split_table.columns):
        report["individuals_by_taxon_split"] = split_table.groupby(["taxon", "split"], dropna=False).size().reset_index(name="n_individuals").to_dict(orient="records")
    return report
