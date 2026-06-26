"""Transfer-scenario utilities for reconstruction experiments.

The reconstruction model is trained on multiple carnivore trajectory sources.
These helpers make the species/habitat relations explicit so that manuscript
results can distinguish, for example:

- same biological species, same habitat;
- same biological species, different habitat;
- different species, same habitat;
- different species, different habitat.

The labels are descriptive analysis labels. They do not change the reconstruction
algorithm unless a notebook explicitly uses them to filter training/test data.
"""
from __future__ import annotations

import pandas as pd

TRANSFER_RELATION_ORDER = [
    "same_species_same_habitat",
    "same_species_different_habitat",
    "different_species_same_habitat",
    "different_species_different_habitat",
    "no_training_source",
]

TRANSFER_RELATION_LABELS = {
    "same_species_same_habitat": "Same species, same habitat",
    "same_species_different_habitat": "Same species, different habitat",
    "different_species_same_habitat": "Different species, same habitat",
    "different_species_different_habitat": "Different species, different habitat",
    "no_training_source": "No training source",
}


def _safe_str(value, default: str = "unknown") -> str:
    if pd.isna(value):
        return default
    s = str(value)
    return s if s and s.lower() != "nan" else default


def classify_transfer_relation(source: dict | pd.Series, target: dict | pd.Series) -> str:
    """Classify the taxonomic/habitat relation from a source row to a target row."""
    s_species = _safe_str(source.get("species_id", source.get("species_group", source.get("taxon", "unknown"))))
    t_species = _safe_str(target.get("species_id", target.get("species_group", target.get("taxon", "unknown"))))
    s_habitat = _safe_str(source.get("habitat_id", source.get("study_system", source.get("dataset", "unknown"))))
    t_habitat = _safe_str(target.get("habitat_id", target.get("study_system", target.get("dataset", "unknown"))))
    same_species = s_species == t_species
    same_habitat = s_habitat == t_habitat
    if same_species and same_habitat:
        return "same_species_same_habitat"
    if same_species and not same_habitat:
        return "same_species_different_habitat"
    if (not same_species) and same_habitat:
        return "different_species_same_habitat"
    return "different_species_different_habitat"


def individual_transfer_units(task_table: pd.DataFrame) -> pd.DataFrame:
    """Return one row per animal with transfer-relevant labels."""
    cols = [
        "dataset", "habitat_id", "study_system", "taxon", "species_common_name",
        "species_id", "species_group", "genus_group", "animal_id", "sex", "age_class",
        "metadata_source", "transfer_unit",
    ]
    keep = [c for c in cols if c in task_table.columns]
    if not keep:
        return pd.DataFrame()
    sort_cols = [c for c in ["species_id", "habitat_id", "dataset", "taxon", "animal_id"] if c in keep]
    return task_table[keep].drop_duplicates().sort_values(sort_cols or keep).reset_index(drop=True)


def task_counts_by_species_habitat(task_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize task counts by taxon/species/habitat/setting/split."""
    group_cols = [
        "dataset", "habitat_id", "study_system", "taxon", "species_id",
        "setting_name", "split",
    ]
    group_cols = [c for c in group_cols if c in task_table.columns]
    if not group_cols:
        return pd.DataFrame()
    return (
        task_table.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_tasks")
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def transfer_scenario_pairs(
    task_table: pd.DataFrame,
    source_split: str = "train",
    target_split: str = "test",
    exclude_same_animal: bool = True,
) -> pd.DataFrame:
    """Return source/target animal pairs and their transfer relation."""
    if "split" not in task_table.columns:
        sources = individual_transfer_units(task_table)
        targets = individual_transfer_units(task_table)
    else:
        sources = individual_transfer_units(task_table[task_table["split"].eq(source_split)])
        targets = individual_transfer_units(task_table[task_table["split"].eq(target_split)])

    rows = []
    for _, s in sources.iterrows():
        for _, t in targets.iterrows():
            same_animal = (
                _safe_str(s.get("dataset")) == _safe_str(t.get("dataset"))
                and _safe_str(s.get("animal_id")) == _safe_str(t.get("animal_id"))
            )
            if exclude_same_animal and same_animal:
                continue
            rows.append({
                "source_dataset": s.get("dataset"),
                "source_habitat_id": s.get("habitat_id"),
                "source_taxon": s.get("taxon"),
                "source_species_id": s.get("species_id"),
                "source_animal_id": s.get("animal_id"),
                "target_dataset": t.get("dataset"),
                "target_habitat_id": t.get("habitat_id"),
                "target_taxon": t.get("taxon"),
                "target_species_id": t.get("species_id"),
                "target_animal_id": t.get("animal_id"),
                "transfer_relation": classify_transfer_relation(s, t),
            })
    return pd.DataFrame(rows)


def transfer_scenario_counts(
    task_table: pd.DataFrame,
    source_split: str = "train",
    target_split: str = "test",
    exclude_same_animal: bool = True,
) -> pd.DataFrame:
    """Count source/target animal-pair scenario types for a split pair."""
    pairs = transfer_scenario_pairs(task_table, source_split, target_split, exclude_same_animal)
    if pairs.empty:
        return pairs
    return (
        pairs.groupby([
            "transfer_relation", "source_species_id", "target_species_id",
            "source_habitat_id", "target_habitat_id",
        ], dropna=False)
        .size()
        .reset_index(name="n_animal_pairs")
        .sort_values(["transfer_relation", "n_animal_pairs"], ascending=[True, False])
        .reset_index(drop=True)
    )


def add_transfer_support_labels(
    task_table: pd.DataFrame,
    source_split: str = "train",
    target_split: str | None = None,
    exclude_same_animal: bool = True,
) -> pd.DataFrame:
    """Annotate target tasks with the training sources available for transfer.

    The output has one row per target task. It includes boolean flags for whether
    at least one training animal is available in each transfer-relation category,
    counts of source animals by category, and a prioritized
    ``best_train_transfer_relation`` label.
    """
    if task_table.empty or "task_uid" not in task_table.columns:
        return pd.DataFrame()
    if "split" in task_table.columns:
        source_tasks = task_table[task_table["split"].eq(source_split)].copy()
        target_tasks = task_table.copy() if target_split is None else task_table[task_table["split"].eq(target_split)].copy()
    else:
        source_tasks = task_table.copy()
        target_tasks = task_table.copy()

    source_units = individual_transfer_units(source_tasks)
    if source_units.empty:
        out = target_tasks[["task_uid"]].drop_duplicates().copy()
        out["best_train_transfer_relation"] = "no_training_source"
        out["n_train_source_animals_total"] = 0
        return out

    target_cols = [
        "task_uid", "dataset", "habitat_id", "study_system", "taxon",
        "species_id", "species_group", "genus_group", "animal_id",
        "setting_name", "split",
    ]
    target_cols = [c for c in target_cols if c in target_tasks.columns]
    targets = target_tasks[target_cols].drop_duplicates("task_uid")

    rows = []
    for _, target in targets.iterrows():
        rel_counts = {rel: 0 for rel in TRANSFER_RELATION_ORDER if rel != "no_training_source"}
        n_total = 0
        for _, source in source_units.iterrows():
            same_animal = (
                _safe_str(source.get("dataset")) == _safe_str(target.get("dataset"))
                and _safe_str(source.get("animal_id")) == _safe_str(target.get("animal_id"))
            )
            if exclude_same_animal and same_animal:
                continue
            rel = classify_transfer_relation(source, target)
            rel_counts[rel] = rel_counts.get(rel, 0) + 1
            n_total += 1
        best = "no_training_source"
        for rel in TRANSFER_RELATION_ORDER:
            if rel == "no_training_source":
                continue
            if rel_counts.get(rel, 0) > 0:
                best = rel
                break
        row = {"task_uid": target.get("task_uid"), "best_train_transfer_relation": best, "n_train_source_animals_total": n_total}
        for rel in TRANSFER_RELATION_ORDER:
            if rel == "no_training_source":
                continue
            safe = rel.replace("different", "diff")
            row[f"n_train_{safe}_animals"] = rel_counts.get(rel, 0)
            row[f"has_train_{safe}"] = rel_counts.get(rel, 0) > 0
        rows.append(row)
    return pd.DataFrame(rows)


def attach_transfer_labels_to_metrics(
    metrics: pd.DataFrame,
    task_table: pd.DataFrame,
    source_split: str = "train",
    target_split: str | None = None,
    exclude_same_animal: bool = True,
) -> pd.DataFrame:
    """Attach transfer-support labels to a task-metric table."""
    if metrics.empty:
        return metrics
    labels = add_transfer_support_labels(task_table, source_split, target_split, exclude_same_animal)
    if labels.empty:
        return metrics
    drop_cols = [c for c in labels.columns if c != "task_uid" and c in metrics.columns]
    return metrics.drop(columns=drop_cols, errors="ignore").merge(labels, on="task_uid", how="left")


def summarize_metrics_by_transfer(
    metrics: pd.DataFrame,
    relation_col: str = "best_train_transfer_relation",
    value_col: str = "ADE",
) -> pd.DataFrame:
    """Summarize method performance by transfer-relation label."""
    required = {"method", relation_col, value_col, "task_uid"}
    if not required.issubset(metrics.columns):
        return pd.DataFrame()
    x = metrics.copy()
    x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
    rows = []
    for (relation, method), g in x.groupby([relation_col, "method"], dropna=False, sort=False):
        vals = g[value_col].dropna()
        rows.append({
            relation_col: relation,
            "method": method,
            "n_tasks": int(g["task_uid"].nunique()),
            f"{value_col}_median": float(vals.median()) if len(vals) else float("nan"),
            f"{value_col}_mean": float(vals.mean()) if len(vals) else float("nan"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order_map = {r: i for i, r in enumerate(TRANSFER_RELATION_ORDER)}
    out["_relation_order"] = out[relation_col].map(order_map).fillna(999).astype(int)
    return out.sort_values(["_relation_order", f"{value_col}_median", "method"]).drop(columns="_relation_order").reset_index(drop=True)
