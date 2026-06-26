"""Motif library construction and nearest-neighbor retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import ReconstructionConfig
from .geometry import (
    denormalize_from_endpoint_frame,
    directness,
    normalize_to_endpoint_frame,
    path_length,
    resample_path_by_index,
)
from .tasks import ReconstructionTask
from .utils import finite_or, make_uid, stable_hash01
from .context import (
    task_context_features, motif_shape_features, build_context_scales, context_match_costs,
)
from .timing import status, ProgressPrinter


@dataclass
class MotifLibrary:
    """Reusable library of normalized LR/HR movement motifs."""

    motif_table: pd.DataFrame
    motif_points: pd.DataFrame
    feature_columns: list[str]
    scaler: StandardScaler
    index: NearestNeighbors
    point_cache: dict[str, np.ndarray] = field(default_factory=dict)
    context_scales: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.point_cache and not self.motif_points.empty:
            self.point_cache = {
                str(uid): g.sort_values("point_order")[["x_norm", "y_norm"]].to_numpy(dtype=float)
                for uid, g in self.motif_points.groupby("motif_id", sort=False)
            }

    def get_frame(self, motif_id: str, n_points: int) -> np.ndarray:
        if not hasattr(self, "point_cache") or self.point_cache is None:
            self.point_cache = {}
        frame = self.point_cache.get(str(motif_id))
        if frame is None:
            g = self.motif_points[self.motif_points["motif_id"].eq(motif_id)].sort_values("point_order")
            frame = g[["x_norm", "y_norm"]].to_numpy(dtype=float)
            self.point_cache[str(motif_id)] = frame
        return resample_path_by_index(frame, n_points)


def _alignment(a: np.ndarray | None, b: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    """Return heading alignment, absolute lateral component, and log step ratio."""
    base = c - b
    base_norm = np.linalg.norm(base)
    if a is None or base_norm <= 1e-9:
        return 0.0, 0.0, 0.0
    vec = b - a
    vec_norm = np.linalg.norm(vec)
    if vec_norm <= 1e-9:
        return 0.0, 0.0, 0.0
    u = base / base_norm
    v = np.array([-u[1], u[0]])
    align = float(np.dot(vec / vec_norm, u))
    lateral_abs = float(abs(np.dot(vec / vec_norm, v)))
    log_ratio = float(np.log((vec_norm + 1.0) / (base_norm + 1.0)))
    return align, lateral_abs, log_ratio


def task_descriptor(task: ReconstructionTask) -> dict[str, float]:
    """Compute deployable coarse-context descriptor for one task."""
    start = task.start_xy
    end = task.end_xy
    disp = float(np.linalg.norm(end - start))
    base_step = disp / max(task.n_points - 1, 1)
    incoming_alignment, incoming_lateral_abs, prev_ratio = _alignment(task.prev_xy, start, end)
    # For outgoing alignment, compare endpoint-to-next vector with start-to-end direction.
    outgoing_alignment = 0.0
    outgoing_lateral_abs = 0.0
    next_ratio = 0.0
    if task.next_xy is not None and disp > 1e-9:
        vec = task.next_xy - end
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 1e-9:
            u = (end - start) / disp
            v = np.array([-u[1], u[0]])
            outgoing_alignment = float(np.dot(vec / vec_norm, u))
            outgoing_lateral_abs = float(abs(np.dot(vec / vec_norm, v)))
            next_ratio = float(np.log((vec_norm + 1.0) / (disp + 1.0)))

    if task.truth_xy is not None:
        endpoint_directness_proxy = finite_or(directness(task.truth_xy), 1.0)
    else:
        endpoint_directness_proxy = 1.0

    out = {
        "n_points_scaled": float(task.n_points) / 20.0,
        "log_coarse_dt": float(np.log1p(task.coarse_dt_min)),
        "log_fine_dt": float(np.log1p(task.fine_dt_min)),
        "log_displacement": float(np.log1p(disp)),
        "log_base_step": float(np.log1p(base_step)),
        "endpoint_directness_proxy": endpoint_directness_proxy,
        "prev_step_log_ratio": prev_ratio,
        "next_step_log_ratio": next_ratio,
        "has_prev": 1.0 if task.prev_xy is not None else 0.0,
        "has_next": 1.0 if task.next_xy is not None else 0.0,
        "incoming_alignment": incoming_alignment,
        "outgoing_alignment": outgoing_alignment,
        "incoming_lateral_abs": incoming_lateral_abs,
        "outgoing_lateral_abs": outgoing_lateral_abs,
    }
    # Add deployable temporal/demographic context to the retrieval descriptor.
    # Environmental layers have heterogeneous names across study systems and are
    # therefore handled as a separate mismatch cost in retrieve_motifs().
    for k, v in task_context_features(task).items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v) if np.isfinite(float(v)) else 0.0
    return out


def build_motif_library(tasks: Sequence[ReconstructionTask], config: ReconstructionConfig, split: str = "train") -> MotifLibrary:
    """Build a motif library from training tasks with known HR truth paths."""
    rows: list[dict] = []
    point_rows: list[dict] = []
    train_tasks = [t for t in tasks if t.truth_xy is not None]
    if not train_tasks:
        raise ValueError("Motif library requires tasks with high-resolution truth paths.")

    status(f"Building motif library from {len(train_tasks):,} training task(s)")
    # Stable downsampling avoids making a new random library every run.
    if config.max_motifs and len(train_tasks) > config.max_motifs:
        status(f"Downsampling motif library to max_motifs={config.max_motifs:,}")
        train_tasks = sorted(train_tasks, key=lambda t: stable_hash01(t.task_uid))[: config.max_motifs]

    progress = ProgressPrinter("build_motif_library", total=len(train_tasks), every=max(1, min(2000, max(1, len(train_tasks)//10))))
    for i, task in enumerate(train_tasks, start=1):
        motif_id = make_uid("motif", task.task_uid)
        desc = task_descriptor(task)
        row = {
            "motif_id": motif_id,
            "source_task_uid": task.task_uid,
            "dataset": task.dataset,
            "taxon": task.taxon,
            "animal_id": task.animal_id,
            "setting_name": task.setting_name,
            "sex": task.sex,
            "age_class": task.age_class,
            "habitat_id": task.habitat_id,
            "study_system": task.study_system,
            "species_common_name": task.species_common_name,
            "species_id": task.species_id,
            "species_group": task.species_group,
            "genus_group": task.genus_group,
            "transfer_unit": task.transfer_unit,
            "metadata_source": task.metadata_source,
            "coarse_dt_min": task.coarse_dt_min,
            "fine_dt_min": task.fine_dt_min,
            "n_points": task.n_points,
            "displacement_m": float(np.linalg.norm(task.end_xy - task.start_xy)),
            "path_length_m": path_length(task.truth_xy),
            "directness": finite_or(directness(task.truth_xy), np.nan),
        }
        # Deployable context stored with each training motif.  These features
        # are used only to retrieve/score motifs for similar endpoint contexts;
        # they do not reveal the target high-resolution path at prediction time.
        frame = normalize_to_endpoint_frame(task.truth_xy)
        row.update(desc)
        row.update(task_context_features(task))
        row.update(motif_shape_features(task, frame=frame))
        rows.append(row)
        for j, (x_norm, y_norm) in enumerate(frame):
            point_rows.append({
                "motif_id": motif_id,
                "point_order": j,
                "x_norm": float(x_norm),
                "y_norm": float(y_norm),
            })
        progress.update(i, extra=f"latest={task.dataset}/{task.taxon}/{task.setting_name}")

    status("Fitting descriptor scaler and nearest-neighbor index")
    motif_table = pd.DataFrame(rows)
    motif_points = pd.DataFrame(point_rows)
    context_scales = build_context_scales(motif_table)
    feature_columns = [c for c in config.descriptor_features if c in motif_table.columns]
    if not feature_columns:
        raise ValueError("No descriptor columns were available for motif retrieval.")
    X = motif_table[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    index = NearestNeighbors(n_neighbors=min(len(motif_table), max(1, config.retrieve_per_task * 4)), algorithm="auto")
    index.fit(Xs)
    status(f"Motif library ready: {len(motif_table):,} motifs; {len(motif_points):,} normalized points")
    return MotifLibrary(motif_table=motif_table, motif_points=motif_points, feature_columns=feature_columns, scaler=scaler, index=index, context_scales=context_scales)


def retrieve_motifs(task: ReconstructionTask, library: MotifLibrary, config: ReconstructionConfig) -> pd.DataFrame:
    """Retrieve and rank candidate motifs for one prediction task."""
    desc = task_descriptor(task)
    X = pd.DataFrame([desc])[library.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    Xs = library.scaler.transform(X)
    mult = int(getattr(config, "context_retrieval_multiplier", 8)) if getattr(config, "use_contextual_retrieval", True) else 4
    k = min(len(library.motif_table), max(config.retrieve_per_task * max(mult, 1), config.retrieve_per_task))
    distances, indices = library.index.kneighbors(Xs, n_neighbors=k, return_distance=True)
    hits = library.motif_table.iloc[indices[0]].copy().reset_index(drop=True)
    hits["descriptor_distance"] = distances[0]
    hits["taxon_penalty"] = np.where(hits["taxon"].astype(str).eq(str(task.taxon)), 0.0, config.taxon_penalty)
    hits["dataset_penalty"] = np.where(hits["dataset"].astype(str).eq(str(task.dataset)), 0.0, config.dataset_penalty)
    if getattr(config, "use_contextual_retrieval", True):
        ctx_rows = [context_match_costs(task, r, getattr(library, "context_scales", {})) for _, r in hits.iterrows()]
        ctx = pd.DataFrame(ctx_rows)
        if not ctx.empty:
            hits = pd.concat([hits.reset_index(drop=True), ctx.reset_index(drop=True)], axis=1)
    for c in ["context_temporal_cost", "context_environment_cost", "context_demographic_cost", "context_n_environment_matches", "context_total_cost"]:
        if c not in hits.columns:
            hits[c] = 0.0
    hits["retrieval_score"] = (
        config.descriptor_weight * hits["descriptor_distance"].astype(float)
        + hits["taxon_penalty"].astype(float)
        + hits["dataset_penalty"].astype(float)
        + getattr(config, "context_temporal_weight", 0.0) * hits["context_temporal_cost"].astype(float)
        + getattr(config, "context_environment_weight", 0.0) * hits["context_environment_cost"].astype(float)
        + getattr(config, "context_demographic_weight", 0.0) * hits["context_demographic_cost"].astype(float)
    )
    hits = hits.sort_values("retrieval_score", kind="mergesort").head(config.retrieve_per_task).reset_index(drop=True)
    hits["retrieval_rank"] = np.arange(1, len(hits) + 1)
    return hits


def motif_to_path(task: ReconstructionTask, library: MotifLibrary, motif_id: str, lateral_scale: float = 1.0, mirror: bool = False) -> np.ndarray:
    """Transfer a normalized motif to a new endpoint pair."""
    frame = library.get_frame(motif_id, task.n_points)
    return denormalize_from_endpoint_frame(frame, task.start_xy, task.end_xy, lateral_scale=lateral_scale, mirror=mirror)
