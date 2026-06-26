"""Construction of low-resolution/high-resolution reconstruction tasks."""
from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ReconstructionConfig
from .data import infer_fine_interval_min
from .geometry import directness, displacement, path_length
from .utils import make_uid, stable_hash01
from .timing import status, ProgressPrinter
from .environment import environmental_columns


@dataclass
class ReconstructionTask:
    """In-memory representation of one endpoint-conditioned gap."""

    task_uid: str
    dataset: str
    taxon: str
    animal_id: str
    setting_name: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    coarse_dt_min: float
    fine_dt_min: float
    n_points: int
    start_xy: np.ndarray
    end_xy: np.ndarray
    truth_xy: np.ndarray | None = None
    prev_xy: np.ndarray | None = None
    next_xy: np.ndarray | None = None
    sex: str = "unknown"
    age_class: str = "unknown"
    habitat_id: str = "unknown"
    study_system: str = "unknown"
    species_common_name: str = "unknown"
    species_id: str = "unknown"
    species_group: str = "unknown"
    genus_group: str = "unknown"
    transfer_unit: str = "unknown"
    metadata_source: str = "unknown"
    truth_env: dict[str, np.ndarray] = field(default_factory=dict)

    def to_row(self) -> dict:
        disp = float(np.linalg.norm(self.end_xy - self.start_xy))
        base_step = disp / max(self.n_points - 1, 1)
        return {
            "task_uid": self.task_uid,
            "dataset": self.dataset,
            "taxon": self.taxon,
            "animal_id": self.animal_id,
            "setting_name": self.setting_name,
            "sex": self.sex,
            "age_class": self.age_class,
            "habitat_id": self.habitat_id,
            "study_system": self.study_system,
            "species_common_name": self.species_common_name,
            "species_id": self.species_id,
            "species_group": self.species_group,
            "genus_group": self.genus_group,
            "transfer_unit": self.transfer_unit,
            "metadata_source": self.metadata_source,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "coarse_dt_min": self.coarse_dt_min,
            "fine_dt_min": self.fine_dt_min,
            "n_points": self.n_points,
            "start_x": float(self.start_xy[0]),
            "start_y": float(self.start_xy[1]),
            "end_x": float(self.end_xy[0]),
            "end_y": float(self.end_xy[1]),
            "displacement_m": disp,
            "base_step_m": base_step,
            "truth_path_length_m": path_length(self.truth_xy) if self.truth_xy is not None else np.nan,
            "truth_directness": directness(self.truth_xy) if self.truth_xy is not None else np.nan,
            "has_prev": self.prev_xy is not None,
            "has_next": self.next_xy is not None,
            "prev_x": float(self.prev_xy[0]) if self.prev_xy is not None else np.nan,
            "prev_y": float(self.prev_xy[1]) if self.prev_xy is not None else np.nan,
            "next_x": float(self.next_xy[0]) if self.next_xy is not None else np.nan,
            "next_y": float(self.next_xy[1]) if self.next_xy is not None else np.nan,
        }


def assign_standard_split(task_uids: Iterable[str], train_fraction: float = 0.70, validation_fraction: float = 0.15) -> dict[str, str]:
    """Assign stable train/validation/test split by task UID hash.

    This helper is kept for simple one-off use. The main task builder uses
    :func:`assign_grouped_standard_split` so each dataset/taxon/setting has an
    explicit 70/15/15 split after task capping.
    """
    train_cut = float(train_fraction)
    val_cut = float(train_fraction + validation_fraction)
    split = {}
    for uid in task_uids:
        h = stable_hash01(uid)
        if h < train_cut:
            split[uid] = "train"
        elif h < val_cut:
            split[uid] = "validation"
        else:
            split[uid] = "test"
    return split


def assign_grouped_standard_split(
    task_table: pd.DataFrame,
    group_cols: list[str] | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> pd.Series:
    """Assign a stable 70/15/15 split within each task setting.

    The split is applied *after* any total task cap per dataset/taxon/setting.
    This avoids the earlier behavior where the cap was applied separately to
    train/validation/test and produced oversized validation/test sets.
    """
    if group_cols is None:
        group_cols = ["dataset", "taxon", "setting_name"]
    if task_table.empty:
        return pd.Series(dtype="object")

    split = pd.Series(index=task_table.index, dtype="object")
    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)

    for _, g in task_table.groupby(group_cols, dropna=False, sort=False):
        idx = list(g.index)
        # Stable pseudo-random ordering independent of row order.
        ordered = sorted(idx, key=lambda i: stable_hash01(str(task_table.at[i, "task_uid"])))
        n = len(ordered)
        n_train = int(round(n * train_fraction))
        n_val = int(round(n * validation_fraction))
        # Keep at least one test task when n is large enough, and prevent overflow.
        if n >= 3 and n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        train_idx = ordered[:n_train]
        val_idx = ordered[n_train:n_train + n_val]
        test_idx = ordered[n_train + n_val:]
        split.loc[train_idx] = "train"
        split.loc[val_idx] = "validation"
        split.loc[test_idx] = "test"

    return split


def _parse_task_settings(raw: object, fallback_fine: float, fallback_coarse_intervals: Iterable[int]) -> list[dict]:
    """Return task settings for one animal group.

    If a dataset spec provided explicit settings, they are stored in the loaded
    tracks as JSON. Otherwise we fall back to the global config using the
    inferred fine interval.
    """
    settings: list[dict] = []
    if raw is not None and not pd.isna(raw) and str(raw).strip():
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            parsed = []
        for item in parsed:
            try:
                coarse = float(item["coarse_dt_min"])
                fine = float(item["fine_dt_min"])
            except Exception:
                continue
            if not np.isfinite(coarse) or not np.isfinite(fine) or fine <= 0 or coarse <= 0:
                continue
            if abs(coarse / fine - round(coarse / fine)) > 1e-6:
                continue
            name = item.get("setting_name") or f"{int(round(coarse))}min_to_{int(round(fine))}min"
            settings.append({"coarse_dt_min": coarse, "fine_dt_min": fine, "setting_name": name})

    if settings:
        # Drop duplicates while preserving order.
        seen = set()
        out = []
        for item in settings:
            key = (round(item["coarse_dt_min"], 6), round(item["fine_dt_min"], 6), item["setting_name"])
            if key not in seen:
                out.append(item)
                seen.add(key)
        return out

    if not np.isfinite(fallback_fine) or fallback_fine <= 0:
        return []
    out = []
    for coarse in fallback_coarse_intervals:
        coarse = float(coarse)
        if coarse <= 0 or coarse < fallback_fine:
            continue
        if abs(coarse / fallback_fine - round(coarse / fallback_fine)) > 1e-6:
            continue
        out.append({
            "coarse_dt_min": coarse,
            "fine_dt_min": float(fallback_fine),
            "setting_name": f"{int(round(coarse))}min_to_{int(round(fallback_fine))}min",
        })
    return out


def make_tasks_from_tracks(tracks: pd.DataFrame, config: ReconstructionConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[ReconstructionTask]]:
    """Create LR/HR training/evaluation tasks from high-resolution tracks.

    Dataset-specific settings are preferred when present in the loaded tracks.
    This lets the cleaned package reproduce the Chapter 5 design:

    * puma: 60/120/240 min endpoints with 5-min truth;
    * Thailand high-resolution collars: 60 min endpoints with 15-min truth;
    * Thailand one-hour collars: 240 min endpoints with 60-min truth;
    * Olympic cougar/bobcat: 240 min endpoints with 60-min truth.

    The function uses exact timestamp lookup for each requested fine interval.
    This avoids accidental ``60_to_60`` or ``120_to_60`` tasks and also allows
    15-min Thailand animals to contribute valid ``240_to_60`` tasks using every
    fourth fix.
    """
    rows: list[dict] = []
    point_rows: list[dict] = []
    tasks: list[ReconstructionTask] = []
    rng = np.random.default_rng(config.random_seed)

    status("Creating endpoint-conditioned LR/HR reconstruction tasks")
    status(
        "Task construction uses dataset-specific settings when available; "
        f"fallback coarse_intervals={tuple(config.coarse_intervals_min)}"
    )
    status(
        f"stride_fraction={config.stride_fraction}, "
        f"max_per_animal_setting={config.max_tasks_per_animal_setting}, "
        f"max_per_dataset_taxon_setting={config.max_tasks_per_dataset_taxon_setting}, "
        f"max_animals_per_setting={getattr(config, 'max_animals_per_dataset_taxon_setting', None)}, "
        f"task_sampling_mode={getattr(config, 'task_sampling_mode', 'full_generalization')}, "
        f"split_strategy={getattr(config, 'split_strategy', 'grouped')}"
    )

    group_cols = ["dataset", "taxon", "animal_id"]
    grouped = list(tracks.groupby(group_cols, dropna=False))
    progress = ProgressPrinter("make_tasks animals", total=len(grouped), every=max(1, min(10, len(grouped))))

    for group_i, ((dataset, taxon, animal_id), g) in enumerate(grouped, start=1):
        g = g.sort_values("time").reset_index(drop=True)
        if len(g) < config.min_points_per_animal:
            progress.update(group_i, extra=f"latest={dataset}/{taxon}/{animal_id}, skipped=too_few_points")
            continue

        inferred_fine = config.fine_interval_min or infer_fine_interval_min(g)
        settings_json = ""
        if "task_settings_json" in g.columns:
            nonempty = g["task_settings_json"].dropna().astype(str)
            nonempty = nonempty[nonempty.str.len() > 0]
            if len(nonempty):
                settings_json = nonempty.iloc[0]
        task_settings = _parse_task_settings(settings_json, inferred_fine, config.coarse_intervals_min)
        if not task_settings:
            progress.update(group_i, extra=f"latest={dataset}/{taxon}/{animal_id}, skipped=no_valid_settings")
            continue

        # Exact timestamp lookup. If duplicated timestamps exist, keep the first
        # occurrence after sorting; duplicated fixes should be cleaned upstream if
        # users want different behavior.
        idx_by_time = {}
        for idx, t in enumerate(g["time"]):
            idx_by_time.setdefault(pd.Timestamp(t), idx)

        animal_task_count_before = len(tasks)
        settings_made: dict[str, int] = {}
        meta = {}
        for meta_col, default in [
            ("sex", "unknown"),
            ("age_class", "unknown"),
            ("habitat_id", str(dataset)),
            ("study_system", str(dataset)),
            ("species_common_name", str(taxon)),
            ("species_id", str(taxon)),
            ("species_group", str(taxon)),
            ("genus_group", str(taxon)),
            ("transfer_unit", f"{taxon}__{dataset}"),
            ("metadata_source", "unknown"),
        ]:
            if meta_col in g.columns and len(g[meta_col].dropna()):
                val = str(g[meta_col].dropna().iloc[0])
                meta[meta_col] = val if val else default
            else:
                meta[meta_col] = default

        for setting in task_settings:
            fine_dt = float(setting["fine_dt_min"])
            coarse_min = float(setting["coarse_dt_min"])
            n_steps = int(round(coarse_min / fine_dt))
            if n_steps < 1 or abs(coarse_min / fine_dt - n_steps) > 1e-6:
                continue
            n_points = n_steps + 1
            if len(g) < n_points:
                continue
            setting_name = str(setting.get("setting_name") or f"{int(round(coarse_min))}min_to_{int(round(fine_dt))}min")

            # Step through possible start rows. The row stride is intentionally
            # conservative; exact timestamp lookup below decides validity.
            stride = max(1, int(round(n_steps * config.stride_fraction)))
            candidates: list[ReconstructionTask] = []

            for start_idx in range(0, len(g), stride):
                t0 = pd.Timestamp(g.loc[start_idx, "time"])
                times = [t0 + pd.Timedelta(minutes=fine_dt * k) for k in range(n_steps + 1)]
                if not all(t in idx_by_time for t in times):
                    continue
                idxs = [idx_by_time[t] for t in times]
                if len(set(idxs)) != len(idxs):
                    continue
                if idxs != sorted(idxs):
                    continue

                segment = g.loc[idxs].copy().reset_index(drop=True)
                actual_dt = (segment["time"].iloc[-1] - segment["time"].iloc[0]).total_seconds() / 60.0
                if not np.isfinite(actual_dt) or abs(actual_dt - coarse_min) > max(fine_dt * 0.25, 1.0):
                    continue

                truth_xy = segment[["x", "y"]].to_numpy(dtype=float)
                if not np.all(np.isfinite(truth_xy)):
                    continue

                env_cols = environmental_columns(segment) if getattr(config, "use_environmental_covariates", True) else []
                truth_env = {c: pd.to_numeric(segment[c], errors="coerce").to_numpy(dtype=float) for c in env_cols}

                # Previous/next context should use the same fine step as the
                # task, not merely the previous row in a denser raw track.
                prev_xy = None
                next_xy = None
                prev_t = times[0] - pd.Timedelta(minutes=fine_dt)
                next_t = times[-1] + pd.Timedelta(minutes=fine_dt)
                if prev_t in idx_by_time:
                    prev_xy = g.loc[idx_by_time[prev_t], ["x", "y"]].to_numpy(dtype=float)
                if next_t in idx_by_time:
                    next_xy = g.loc[idx_by_time[next_t], ["x", "y"]].to_numpy(dtype=float)

                uid = make_uid(dataset, taxon, animal_id, setting_name, times[0])
                task = ReconstructionTask(
                    task_uid=uid,
                    dataset=str(dataset),
                    taxon=str(taxon),
                    animal_id=str(animal_id),
                    setting_name=setting_name,
                    sex=meta.get("sex", "unknown"),
                    age_class=meta.get("age_class", "unknown"),
                    habitat_id=meta.get("habitat_id", str(dataset)),
                    study_system=meta.get("study_system", str(dataset)),
                    species_common_name=meta.get("species_common_name", str(taxon)),
                    species_id=meta.get("species_id", str(taxon)),
                    species_group=meta.get("species_group", meta.get("species_id", str(taxon))),
                    genus_group=meta.get("genus_group", str(taxon)),
                    transfer_unit=meta.get("transfer_unit", f"{meta.get('species_id', str(taxon))}__{meta.get('habitat_id', str(dataset))}"),
                    metadata_source=meta.get("metadata_source", "unknown"),
                    start_time=segment["time"].iloc[0],
                    end_time=segment["time"].iloc[-1],
                    coarse_dt_min=float(actual_dt),
                    fine_dt_min=float(fine_dt),
                    n_points=int(n_points),
                    start_xy=truth_xy[0],
                    end_xy=truth_xy[-1],
                    truth_xy=truth_xy,
                    truth_env=truth_env,
                    prev_xy=prev_xy,
                    next_xy=next_xy,
                )
                candidates.append(task)

            if config.max_tasks_per_animal_setting and len(candidates) > config.max_tasks_per_animal_setting:
                keep = rng.choice(len(candidates), size=config.max_tasks_per_animal_setting, replace=False)
                candidates = [candidates[i] for i in sorted(keep)]

            settings_made[setting_name] = len(candidates)
            for task in candidates:
                rows.append(task.to_row())
                for point_order, (x, y) in enumerate(task.truth_xy):
                    pr = {
                        "task_uid": task.task_uid,
                        "point_order": point_order,
                        "time": task.start_time + pd.to_timedelta(point_order * task.fine_dt_min, unit="min"),
                        "x": float(x),
                        "y": float(y),
                    }
                    for env_name, env_values in task.truth_env.items():
                        if point_order < len(env_values):
                            pr[env_name] = float(env_values[point_order]) if np.isfinite(env_values[point_order]) else np.nan
                    point_rows.append(pr)
                tasks.append(task)

        made_for_animal = len(tasks) - animal_task_count_before
        setting_summary = ", ".join(f"{k}:{v}" for k, v in settings_made.items() if v)
        if not setting_summary:
            setting_summary = "no_tasks"
        progress.update(group_i, extra=f"latest={dataset}/{taxon}/{animal_id}, tasks_added={made_for_animal}, {setting_summary}")

    status(f"Initial task creation finished: {len(rows):,} tasks before dataset/taxon cap")
    task_table = pd.DataFrame(rows)
    if task_table.empty:
        raise ValueError("No reconstruction tasks were created. Check intervals and input schema.")

    # Optional old-paper comparable balancing: retain a small number of animals
    # per dataset/taxon/setting before the total stratum cap. This reproduces
    # the older balanced diagnostic design more closely and prevents the large
    # Thailand archive from dominating the benchmark.
    if getattr(config, "max_animals_per_dataset_taxon_setting", None):
        status("Applying max animals per dataset/taxon/setting before task cap")
        keep_uids = []
        for _, g in task_table.groupby(["dataset", "taxon", "setting_name"], dropna=False, sort=False):
            counts = g.groupby("animal_id")["task_uid"].nunique().reset_index(name="n")
            counts = counts.sort_values(["n", "animal_id"], ascending=[False, True], kind="mergesort")
            keep_animals = set(counts.head(int(config.max_animals_per_dataset_taxon_setting))["animal_id"].astype(str))
            keep_uids.extend(g[g["animal_id"].astype(str).isin(keep_animals)]["task_uid"].tolist())
        keep_uids = set(keep_uids)
        task_table = task_table[task_table["task_uid"].isin(keep_uids)].reset_index(drop=True)
        tasks = [t for t in tasks if t.task_uid in keep_uids]
        point_rows = [r for r in point_rows if r["task_uid"] in keep_uids]

    if config.max_tasks_per_dataset_taxon_setting:
        status("Applying total dataset/taxon/setting task cap before splitting")
        keep_uids = []
        for _, g in task_table.groupby(["dataset", "taxon", "setting_name"], dropna=False, sort=False):
            if len(g) <= config.max_tasks_per_dataset_taxon_setting:
                keep_uids.extend(g["task_uid"].tolist())
            else:
                # Stable random sample within the whole setting, then split.
                keep = g.sample(config.max_tasks_per_dataset_taxon_setting, random_state=config.random_seed)
                keep_uids.extend(keep["task_uid"].tolist())
        keep_uids = set(keep_uids)
        task_table = task_table[task_table["task_uid"].isin(keep_uids)].reset_index(drop=True)
        tasks = [t for t in tasks if t.task_uid in keep_uids]
        point_rows = [r for r in point_rows if r["task_uid"] in keep_uids]

    if getattr(config, "split_strategy", "grouped") == "global_hash":
        status("Assigning stable global hash train/validation/test split")
        split_map = assign_standard_split(task_table["task_uid"].astype(str), config.train_fraction, config.validation_fraction)
        task_table["split"] = task_table["task_uid"].astype(str).map(split_map).values
    else:
        status("Assigning stable train/validation/test split within each dataset/taxon/setting")
        task_table["split"] = assign_grouped_standard_split(
            task_table,
            group_cols=["dataset", "taxon", "setting_name"],
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
        ).values

    task_points = pd.DataFrame(point_rows)
    status(f"Final task table: {len(task_table):,} tasks; truth point rows: {len(task_points):,}")
    status("Tasks by dataset/taxon/setting:")
    try:
        summary = task_table.groupby(["dataset", "taxon", "setting_name", "split"], dropna=False).size().reset_index(name="n_tasks")
        for _, row in summary.iterrows():
            status(f"  {row['dataset']} | {row['taxon']} | {row['setting_name']} | {row['split']}: {int(row['n_tasks'])}")
    except Exception:
        pass
    return task_table.reset_index(drop=True), task_points.reset_index(drop=True), tasks


def tasks_from_tables(task_table: pd.DataFrame, task_points: pd.DataFrame | None = None) -> list[ReconstructionTask]:
    """Rebuild task objects from saved metadata and optional truth-point table."""
    point_lookup = {}
    env_lookup = {}
    if task_points is not None and not task_points.empty:
        env_cols = environmental_columns(task_points)
        for uid, g in task_points.groupby("task_uid"):
            gg = g.sort_values("point_order")
            point_lookup[uid] = gg[["x", "y"]].to_numpy(dtype=float)
            env_lookup[uid] = {c: pd.to_numeric(gg[c], errors="coerce").to_numpy(dtype=float) for c in env_cols}
    tasks = []
    for _, r in task_table.iterrows():
        uid = str(r["task_uid"])
        prev_xy = None
        next_xy = None
        if np.isfinite(r.get("prev_x", np.nan)) and np.isfinite(r.get("prev_y", np.nan)):
            prev_xy = np.array([float(r["prev_x"]), float(r["prev_y"])])
        if np.isfinite(r.get("next_x", np.nan)) and np.isfinite(r.get("next_y", np.nan)):
            next_xy = np.array([float(r["next_x"]), float(r["next_y"])])
        tasks.append(ReconstructionTask(
            task_uid=uid,
            dataset=str(r["dataset"]),
            taxon=str(r["taxon"]),
            animal_id=str(r["animal_id"]),
            setting_name=str(r["setting_name"]),
            sex=str(r.get("sex", "unknown")),
            age_class=str(r.get("age_class", "unknown")),
            habitat_id=str(r.get("habitat_id", r.get("dataset", "unknown"))),
            study_system=str(r.get("study_system", r.get("dataset", "unknown"))),
            species_common_name=str(r.get("species_common_name", r.get("taxon", "unknown"))),
            species_id=str(r.get("species_id", r.get("species_group", r.get("taxon", "unknown")))),
            species_group=str(r.get("species_group", r.get("species_id", r.get("taxon", "unknown")))),
            genus_group=str(r.get("genus_group", r.get("taxon", "unknown"))),
            transfer_unit=str(r.get("transfer_unit", f"{r.get('species_id', r.get('taxon', 'unknown'))}__{r.get('habitat_id', r.get('dataset', 'unknown'))}")),
            metadata_source=str(r.get("metadata_source", "unknown")),
            start_time=pd.to_datetime(r["start_time"]),
            end_time=pd.to_datetime(r["end_time"]),
            coarse_dt_min=float(r["coarse_dt_min"]),
            fine_dt_min=float(r["fine_dt_min"]),
            n_points=int(r["n_points"]),
            start_xy=np.array([float(r["start_x"]), float(r["start_y"])]),
            end_xy=np.array([float(r["end_x"]), float(r["end_y"])]),
            truth_xy=point_lookup.get(uid),
            truth_env=env_lookup.get(uid, {}),
            prev_xy=prev_xy,
            next_xy=next_xy,
        ))
    return tasks


def make_prediction_tasks(
    coarse_track: pd.DataFrame,
    fine_interval_min: float,
    dataset: str = "user",
    taxon: str = "unknown",
    animal_id: str = "animal",
    sex: str = "unknown",
    age_class: str = "unknown",
    habitat_id: str | None = None,
    species_id: str | None = None,
) -> list[ReconstructionTask]:
    """Create prediction-only tasks from a coarse trajectory.

    ``coarse_track`` must contain ``time``, ``x``, and ``y`` columns.  Consecutive
    rows become endpoint-conditioned gaps.
    """
    df = coarse_track.copy()
    habitat_id = habitat_id or dataset
    species_id = species_id or {
        "puma": "puma_concolor",
        "cougar": "puma_concolor",
        "bobcat": "lynx_rufus",
        "tiger": "panthera_tigris",
        "leopard": "panthera_pardus",
    }.get(str(taxon), str(taxon))
    genus_group = {
        "puma": "puma", "cougar": "puma", "bobcat": "lynx",
        "tiger": "panthera", "leopard": "panthera",
    }.get(str(taxon), str(taxon))
    transfer_unit = f"{species_id}__{habitat_id}"
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    tasks: list[ReconstructionTask] = []
    for i in range(len(df) - 1):
        start = df.loc[i, ["x", "y"]].to_numpy(dtype=float)
        end = df.loc[i + 1, ["x", "y"]].to_numpy(dtype=float)
        dt = (df.loc[i + 1, "time"] - df.loc[i, "time"]).total_seconds() / 60.0
        if not np.isfinite(dt) or dt <= 0:
            continue
        n_points = int(round(dt / fine_interval_min)) + 1
        if n_points < 2:
            n_points = 2
        prev_xy = df.loc[i - 1, ["x", "y"]].to_numpy(dtype=float) if i > 0 else None
        next_xy = df.loc[i + 2, ["x", "y"]].to_numpy(dtype=float) if i + 2 < len(df) else None
        setting = f"{int(round(dt))}min_to_{int(round(fine_interval_min))}min"
        uid = make_uid(dataset, taxon, animal_id, setting, df.loc[i, "time"], i)
        tasks.append(ReconstructionTask(
            task_uid=uid,
            dataset=dataset,
            taxon=taxon,
            animal_id=animal_id,
            setting_name=setting,
            sex=sex,
            age_class=age_class,
            habitat_id=habitat_id,
            study_system=habitat_id,
            species_common_name=taxon,
            species_id=species_id,
            species_group=species_id,
            genus_group=genus_group,
            transfer_unit=transfer_unit,
            metadata_source="user_input",
            start_time=df.loc[i, "time"],
            end_time=df.loc[i + 1, "time"],
            coarse_dt_min=float(dt),
            fine_dt_min=float(fine_interval_min),
            n_points=int(n_points),
            start_xy=start,
            end_xy=end,
            truth_xy=None,
            prev_xy=prev_xy,
            next_xy=next_xy,
        ))
    return tasks
