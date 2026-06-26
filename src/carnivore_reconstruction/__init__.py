"""Clean motif-based carnivore trajectory reconstruction package."""
from .config import DatasetSpec, ReconstructionConfig
from .data import load_dataset, load_datasets, standardize_track_table
from .model import MotifReconstructionModel, ReconstructionResult
from .tasks import make_tasks_from_tracks, make_prediction_tasks, tasks_from_tables
from .proposed import evaluate_proposed_methods_for_tasks, make_guarded_setting_selection

__all__ = [
    "DatasetSpec",
    "ReconstructionConfig",
    "load_dataset",
    "load_datasets",
    "standardize_track_table",
    "MotifReconstructionModel",
    "ReconstructionResult",
    "make_tasks_from_tracks",
    "make_prediction_tasks",
    "tasks_from_tables",
    "evaluate_proposed_methods_for_tasks",
    "make_guarded_setting_selection",
    "classify_transfer_relation",
    "individual_transfer_units",
    "task_counts_by_species_habitat",
    "transfer_scenario_pairs",
    "transfer_scenario_counts",
    "add_transfer_support_labels",
    "attach_transfer_labels_to_metrics",
    "summarize_metrics_by_transfer",
]

# Plotting utilities are available from carnivore_reconstruction.plotting.

from .transfer import (
    classify_transfer_relation,
    individual_transfer_units,
    task_counts_by_species_habitat,
    transfer_scenario_pairs,
    transfer_scenario_counts,
    add_transfer_support_labels,
    attach_transfer_labels_to_metrics,
    summarize_metrics_by_transfer,
)

from .environment import sample_rasters_at_xy, resolve_raster_paths, environmental_columns
