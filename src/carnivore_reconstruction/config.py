"""Configuration objects for motif-based carnivore trajectory reconstruction.

The package is intentionally small and notebook-friendly.  Users can keep the
configuration in a notebook cell, but the dataclasses here make the expected
fields explicit and help avoid version-specific global variables.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional, Any


@dataclass
class DatasetSpec:
    """Description of one high-resolution tracking dataset.

    Parameters
    ----------
    path:
        CSV file containing high-resolution fixes.
    dataset:
        Short dataset name, e.g. ``"SantaCruz_puma"``.
    taxon:
        Species/group label, e.g. ``"puma"`` or ``"tiger"``.
    id_col, time_col, x_col, y_col:
        Column names in ``path``.  If projected x/y columns are not available,
        provide ``lon_col``, ``lat_col``, and ``epsg`` instead.
    """

    path: str | Path
    dataset: str
    taxon: str
    id_col: str = "animal_id"
    time_col: str = "timestamp"
    x_col: Optional[str] = "x"
    y_col: Optional[str] = "y"
    lon_col: Optional[str] = None
    lat_col: Optional[str] = None
    epsg: Optional[int] = None
    extra_paths: list[str | Path] = field(default_factory=list)
    env_cols: list[str] = field(default_factory=list)
    # Optional raster-based environmental covariates. Keys are output column
    # names and values are raster paths. Paths may be absolute or relative to
    # the project root / data/raw directory. When present, rasters are sampled
    # onto observed fixes at load time and can also be used to evaluate
    # reconstructed candidate-path exposure.
    raster_paths: dict[str, str | Path] = field(default_factory=dict)
    raster_dir: Optional[str | Path] = None
    # CRS/EPSG of rasters for this dataset. If omitted, the dataset ``epsg`` is
    # used. This is intentionally explicit because some local PROJ installs fail
    # to read EPSG metadata from GeoTIFFs even when the raster and track x/y are
    # already in the same projected CRS.
    raster_epsg: Optional[int] = None


    # Optional dataset-specific task controls. These let one dataset use only
    # meaningful reconstruction settings instead of the global defaults. Each
    # setting is a small dict, for example:
    #   {"coarse_dt_min": 60, "fine_dt_min": 5}
    # Animal-specific settings override ``task_settings`` for matching IDs.
    task_settings: list[dict[str, Any]] = field(default_factory=list)
    animal_task_settings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Optional per-animal taxon relabeling. Useful for combined tiger/leopard
    # CSVs where the raw table has collar IDs but no clean species column.
    taxon_by_animal_id: dict[str, str] = field(default_factory=dict)

    # Optional individual metadata table. The loader detects columns when exact
    # names are not provided. This is used only for metadata propagation and
    # stratified analysis by default; it is not used as a reconstruction feature.
    metadata_path: Optional[str | Path] = None
    metadata_id_col: Optional[str] = None
    metadata_code_col: Optional[str] = None
    metadata_sex_col: Optional[str] = None
    metadata_age_col: Optional[str] = None
    metadata_species_col: Optional[str] = None

    def to_dict(self) -> dict:
        out = asdict(self)
        out["path"] = str(out["path"])
        out["extra_paths"] = [str(p) for p in out.get("extra_paths", [])]
        if out.get("metadata_path") is not None:
            out["metadata_path"] = str(out["metadata_path"])
        if out.get("raster_dir") is not None:
            out["raster_dir"] = str(out["raster_dir"])
        if out.get("raster_paths"):
            out["raster_paths"] = {k: str(v) for k, v in out["raster_paths"].items()}
        return out


@dataclass
class ReconstructionConfig:
    """Core method settings.

    The defaults are deliberately conservative and fast.  They correspond to the
    cleaned backbone: paired LR/HR motif retrieval + endpoint transformation +
    guarded representative selection.
    """

    random_seed: int = 42
    output_dir: str | Path = "outputs"

    # Training-task construction.
    coarse_intervals_min: tuple[int, ...] = (60, 120, 240)
    fine_interval_min: Optional[float] = None
    stride_fraction: float = 1.0
    max_tasks_per_dataset_taxon_setting: Optional[int] = None
    max_tasks_per_animal_setting: Optional[int] = None
    min_points_per_animal: int = 5

    # Split. The default grouped 70/15/15 split is appropriate for full
    # generalization tests. The legacy-comparable mode below reproduces the
    # older balanced task design more closely: cap tasks within
    # each dataset/taxon/setting, then assign the split by a global stable hash.
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    task_sampling_mode: str = "full_generalization"  # or "legacy_comparable"
    split_strategy: str = "grouped"  # "grouped" or "global_hash"
    max_animals_per_dataset_taxon_setting: Optional[int] = None


    # Environmental/raster support. If True, any environmental columns or sampled
    # rasters are propagated to task points and used for environmental exposure
    # diagnostics when predicted-path rasters are available. These features are
    # recorded for analysis; the main pretrained motif selector remains fully
    # deployable without hidden truth.
    use_environmental_covariates: bool = True
    evaluate_environmental_exposure: bool = True

    # Motif library / retrieval.
    max_motifs: int = 25_000
    retrieve_per_task: int = 20
    descriptor_features: tuple[str, ...] = (
        "n_points_scaled",
        "log_coarse_dt",
        "log_fine_dt",
        "log_displacement",
        "log_base_step",
        "endpoint_directness_proxy",
        "prev_step_log_ratio",
        "next_step_log_ratio",
        "has_prev",
        "has_next",
        "incoming_alignment",
        "outgoing_alignment",
        "incoming_lateral_abs",
        "outgoing_lateral_abs",
        # Deployable temporal context. Environmental context is handled by a
        # separate robust mismatch cost because layer names differ by study.
        "start_hour_sin",
        "start_hour_cos",
        "mid_hour_sin",
        "mid_hour_cos",
        "month_sin",
        "month_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "context_sex_female",
        "context_sex_male",
        "context_age_adult",
        "context_age_subadult",
    )
    descriptor_weight: float = 0.85
    taxon_penalty: float = 0.12
    dataset_penalty: float = 0.05

    # Context-aware retrieval/scoring. These terms are deployable: they use
    # endpoint environmental values, timestamp, season, and individual metadata,
    # and they compare target tasks against training motifs. They do not use the
    # hidden high-resolution target path.
    use_contextual_retrieval: bool = True
    context_temporal_weight: float = 0.12
    context_environment_weight: float = 0.22
    context_demographic_weight: float = 0.08
    context_direction_weight: float = 0.25
    context_efficiency_weight: float = 0.30
    context_source_shape_weight: float = 0.25
    context_max_preferred_path_ratio: float = 2.25
    context_retrieval_multiplier: int = 8

    # Candidate generation.  Keep this small for interactive runtime.
    # The proposed method uses a proposed-only candidate bank: internal direct/heading motifs
    # plus retrieved LR/HR motif transfers. Baselines are evaluated separately.
    include_mirror: bool = True
    lateral_scales: tuple[float, ...] = (0.75, 1.0, 1.2)
    include_direct_identity_motif: bool = True
    include_heading_continuity_motif: bool = True
    # Backward-compatible switches; no longer mixed into the proposed bank.
    include_linear_candidate: bool = False
    include_heading_candidate: bool = False
    max_candidates_per_task: int = 45

    # Scoring and guards.
    step_capacity_quantile: float = 0.90
    step_capacity_min_multiplier: float = 1.10
    max_path_ratio: float = 3.0
    max_step_violation_fraction: float = 0.20
    representative_top_k: int = 20
    output_top_k: int = 5
    # K20 representative beta used for user-facing reconstruction.
    # Paper benchmarking also evaluates the representative beta grid 0.25/0.35/0.50/0.75.
    softmax_beta: float = 0.50

    # Runtime behavior.
    # Runtime parallelism. Use 0/None/negative for automatic threads.
    # Auto mode uses max(1, os.cpu_count() - 1), capped to avoid saturating the machine.
    n_jobs: int | None = 0
    parallel_threshold_tasks: int = 2
    save_candidate_paths: bool = False

    def output_path(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> dict:
        out = asdict(self)
        out["output_dir"] = str(out["output_dir"])
        out["coarse_intervals_min"] = list(out["coarse_intervals_min"])
        out["descriptor_features"] = list(out["descriptor_features"])
        out["lateral_scales"] = list(out["lateral_scales"])
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "ReconstructionConfig":
        data = dict(data)
        for key in ["coarse_intervals_min", "descriptor_features", "lateral_scales"]:
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)
