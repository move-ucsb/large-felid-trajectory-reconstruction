"""Candidate path generation for the cleaned reconstruction backbone."""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd

from .config import ReconstructionConfig
from .geometry import brownian_bridge_path, hermite_path, linear_path, rtg_bridge_path
from .library import MotifLibrary, motif_to_path, retrieve_motifs
from .tasks import ReconstructionTask
from .utils import make_uid
from .transfer import classify_transfer_relation


def generate_candidates_for_task(task: ReconstructionTask, library: MotifLibrary, config: ReconstructionConfig) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    """Generate proposed candidates for one task.

    This is intentionally different from the earlier simplified package: paper
    baselines are *not* inserted into the proposed candidate bank.  The bank is
    proposed-method ``motif_timegeo_sr`` only, with two internal proposed fallback
    motifs (direct identity and heading continuity) plus retrieved LR/HR motif
    transfers. Baselines are generated only by :func:`generate_baseline_paths`.
    """
    t0 = time.perf_counter()
    retrieve_start = time.perf_counter()
    motifs = retrieve_motifs(task, library, config)
    retrieve_seconds = time.perf_counter() - retrieve_start

    path_start = time.perf_counter()
    rows: list[dict] = []
    paths: dict[str, np.ndarray] = {}

    def add_candidate(style: str, path: np.ndarray, **meta) -> None:
        cid = make_uid(task.task_uid, "motif_timegeo_sr", len(rows), meta.get("motif_id", ""), meta.get("mirror", ""), meta.get("lateral_scale", ""), style)
        paths[cid] = path
        row = {
            "task_uid": task.task_uid,
            "candidate_id": cid,
            "origin": "motif_timegeo_sr",
            "method_family": "proposed_motif",
            "style": style,
            "candidate_order": len(rows),
        }
        row.update(meta)
        rows.append(row)

    # The proposed method internal direct motif: belongs to the proposed bank, not to the
    # external linear baseline. It allows the proposed method to choose a
    # conservative near-linear path when the retrieved motif bank is uncertain.
    if getattr(config, "include_direct_identity_motif", True):
        add_candidate(
            "direct_identity",
            linear_path(task.start_xy, task.end_xy, task.n_points),
            motif_id="direct_identity",
            retrieval_rank=0,
            retrieval_score=0.12,
            descriptor_distance=0.0,
            lateral_scale=0.0,
            mirror=False,
            source_dataset="proposed_internal",
            source_taxon=str(task.taxon),
            source_animal_id=str(task.animal_id),
            source_habitat_id=str(task.habitat_id),
            source_study_system=str(task.study_system),
            source_species_id=str(task.species_id),
            source_genus_group=str(task.genus_group),
            source_transfer_relation="proposed_internal",
            source_path_ratio=1.0,
            source_directness=1.0,
            source_lateral_mean_norm=0.0,
            source_lateral_sign=0.0,
            context_temporal_cost=0.0,
            context_environment_cost=0.0,
            context_demographic_cost=0.0,
            context_total_cost=0.0,
            context_n_environment_matches=0.0,
            candidate_direct_identity_flag=1,
            candidate_heading_continuity_flag=0,
        )

    # The proposed method internal heading-continuity motif: also part of the proposed method,
    # not the external heading-Hermite baseline.
    if getattr(config, "include_heading_continuity_motif", True) and (task.prev_xy is not None or task.next_xy is not None):
        add_candidate(
            "heading_continuity",
            hermite_path(task.start_xy, task.end_xy, task.n_points, prev_xy=task.prev_xy, next_xy=task.next_xy),
            motif_id="heading_continuity",
            retrieval_rank=0,
            retrieval_score=0.18,
            descriptor_distance=0.0,
            lateral_scale=1.0,
            mirror=False,
            source_dataset="proposed_internal",
            source_taxon=str(task.taxon),
            source_animal_id=str(task.animal_id),
            source_habitat_id=str(task.habitat_id),
            source_study_system=str(task.study_system),
            source_species_id=str(task.species_id),
            source_genus_group=str(task.genus_group),
            source_transfer_relation="proposed_internal",
            source_path_ratio=np.nan,
            source_directness=np.nan,
            source_lateral_mean_norm=0.0,
            source_lateral_sign=0.0,
            context_temporal_cost=0.0,
            context_environment_cost=0.0,
            context_demographic_cost=0.0,
            context_total_cost=0.0,
            context_n_environment_matches=0.0,
            candidate_direct_identity_flag=0,
            candidate_heading_continuity_flag=1,
        )

    # Retrieved paired LR/HR motif transfers.
    mirror_values = [False, True] if config.include_mirror else [False]
    max_candidates = int(getattr(config, "max_candidates_per_task", 45))
    for _, hit in motifs.iterrows():
        for mirror in mirror_values:
            for lateral_scale in config.lateral_scales:
                if len(rows) >= max_candidates:
                    break
                path = motif_to_path(task, library, str(hit["motif_id"]), lateral_scale=float(lateral_scale), mirror=bool(mirror))
                add_candidate(
                    "retrieved_motif",
                    path,
                    motif_id=str(hit["motif_id"]),
                    source_task_uid=str(hit["source_task_uid"]),
                    retrieval_rank=int(hit["retrieval_rank"]),
                    retrieval_score=float(hit["retrieval_score"]),
                    descriptor_distance=float(hit["descriptor_distance"]),
                    lateral_scale=float(lateral_scale),
                    mirror=bool(mirror),
                    source_dataset=str(hit.get("dataset", "unknown")),
                    source_taxon=str(hit.get("taxon", "unknown")),
                    source_animal_id=str(hit.get("animal_id", "unknown")),
                    source_habitat_id=str(hit.get("habitat_id", "unknown")),
                    source_study_system=str(hit.get("study_system", "unknown")),
                    source_species_id=str(hit.get("species_id", hit.get("species_group", hit.get("taxon", "unknown")))),
                    source_genus_group=str(hit.get("genus_group", "unknown")),
                    source_transfer_relation=classify_transfer_relation(hit, task.to_row()),
                    source_path_ratio=float(hit.get("motif_path_ratio", np.nan)),
                    source_directness=float(hit.get("motif_directness", np.nan)),
                    source_lateral_mean_norm=float(hit.get("motif_lateral_mean_norm", 0.0)),
                    source_lateral_sign=float(hit.get("motif_lateral_sign", 0.0)) * (-1.0 if bool(mirror) else 1.0),
                    source_lateral_abs_mean_norm=float(hit.get("motif_lateral_abs_mean_norm", 0.0)),
                    source_lateral_abs_max_norm=float(hit.get("motif_lateral_abs_max_norm", 0.0)),
                    context_temporal_cost=float(hit.get("context_temporal_cost", 0.0)),
                    context_environment_cost=float(hit.get("context_environment_cost", 0.0)),
                    context_demographic_cost=float(hit.get("context_demographic_cost", 0.0)),
                    context_total_cost=float(hit.get("context_total_cost", 0.0)),
                    context_n_environment_matches=float(hit.get("context_n_environment_matches", 0.0)),
                    candidate_direct_identity_flag=0,
                    candidate_heading_continuity_flag=0,
                )
            if len(rows) >= max_candidates:
                break
        if len(rows) >= max_candidates:
            break

    # If retrieval is empty and internal fallbacks were disabled, add one
    # proposed fallback so downstream the proposed method tables remain full coverage.
    if not rows:
        add_candidate(
            "fallback_linear",
            linear_path(task.start_xy, task.end_xy, task.n_points),
            motif_id="fallback_linear",
            retrieval_rank=0,
            retrieval_score=1.0,
            descriptor_distance=1.0,
            lateral_scale=1.0,
            mirror=False,
            source_dataset="proposed_internal",
            source_taxon=str(task.taxon),
            source_animal_id=str(task.animal_id),
            source_habitat_id=str(task.habitat_id),
            source_study_system=str(task.study_system),
            source_species_id=str(task.species_id),
            source_genus_group=str(task.genus_group),
            source_transfer_relation="proposed_internal",
            source_path_ratio=1.0,
            source_directness=1.0,
            source_lateral_mean_norm=0.0,
            source_lateral_sign=0.0,
            context_temporal_cost=0.0,
            context_environment_cost=0.0,
            context_demographic_cost=0.0,
            context_total_cost=0.0,
            context_n_environment_matches=0.0,
            candidate_direct_identity_flag=1,
            candidate_heading_continuity_flag=0,
        )

    # De-duplicate near-identical proposed paths while preserving the proposed method order.
    seen = set()
    keep_rows = []
    keep_paths = {}
    for row in rows:
        cid = str(row["candidate_id"])
        path = paths[cid]
        key = (str(row.get("style", "")), tuple(np.round(path.reshape(-1), 3)))
        if key in seen:
            continue
        seen.add(key)
        row = dict(row)
        row["candidate_order"] = len(keep_rows)
        keep_rows.append(row)
        keep_paths[cid] = path

    path_seconds = time.perf_counter() - path_start
    table = pd.DataFrame(keep_rows)
    timing = {
        "task_uid": task.task_uid,
        "n_candidates": len(table),
        "retrieve_seconds": retrieve_seconds,
        "generate_path_seconds": path_seconds,
        "total_candidate_seconds": time.perf_counter() - t0,
    }
    return table, keep_paths, timing

def generate_baseline_paths(task: ReconstructionTask, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate simple baseline paths for benchmark notebooks."""
    return {
        "linear": linear_path(task.start_xy, task.end_xy, task.n_points),
        "heading_hermite": hermite_path(task.start_xy, task.end_xy, task.n_points, prev_xy=task.prev_xy, next_xy=task.next_xy),
        "brownian_bridge": brownian_bridge_path(task.start_xy, task.end_xy, task.n_points, seed=seed),
        "rtg_bridge": rtg_bridge_path(task.start_xy, task.end_xy, task.n_points),
    }
