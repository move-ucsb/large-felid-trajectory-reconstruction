"""High-level model interface for clean notebooks and user-facing scripts."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd

from .candidates import generate_candidates_for_task
from .config import ReconstructionConfig
from .library import MotifLibrary, build_motif_library
from .ranking import build_movement_priors, score_candidates, select_paths
from .tasks import ReconstructionTask, make_prediction_tasks
from .timing import TimerLog, status, ProgressPrinter
from .utils import write_json, write_table


@dataclass
class ReconstructionResult:
    """Output bundle returned by prediction and benchmark helpers."""

    selected_table: pd.DataFrame
    path_table: pd.DataFrame
    task_summary: pd.DataFrame
    runtime_table: pd.DataFrame
    scored_candidates: pd.DataFrame | None = None


class MotifReconstructionModel:
    """Pretrained motif-retrieval trajectory reconstruction model.

    The model can be fitted once from high-resolution tracks/tasks and then used
    repeatedly for fast endpoint-conditioned reconstruction of sparse tracks.
    """

    def __init__(self, config: ReconstructionConfig | None = None) -> None:
        self.config = config or ReconstructionConfig()
        self.library: MotifLibrary | None = None
        self.movement_priors: pd.DataFrame | None = None
        self.task_table: pd.DataFrame | None = None
        self.metadata: dict = {}
        self.environment_raster_paths: dict = {}
        self.environment_raster_paths_by_dataset: dict = {}
        self.environment_epsg_by_dataset: dict = {}
        self.v31_artifacts: dict = {}
        self.ecological_artifacts: dict = {}

    def fit(self, tasks: Sequence[ReconstructionTask], task_table: pd.DataFrame | None = None, train_split: str = "train", timer: TimerLog | None = None) -> "MotifReconstructionModel":
        """Fit the pretrained model from LR/HR reconstruction tasks."""
        timer = timer or TimerLog()
        if task_table is not None and "split" in task_table.columns:
            train_uids = set(task_table.loc[task_table["split"].eq(train_split), "task_uid"].astype(str))
            train_tasks = [t for t in tasks if t.task_uid in train_uids]
        else:
            train_tasks = list(tasks)
        if not train_tasks:
            raise ValueError("No training tasks were available for model fitting.")

        status(f"Model fitting uses {len(train_tasks):,} task(s) from split='{train_split}'")
        with timer.step("build_motif_library", n_tasks=len(train_tasks)):
            self.library = build_motif_library(train_tasks, self.config, split=train_split)

        with timer.step("build_movement_priors", n_tasks=len(train_tasks)):
            self.movement_priors = build_movement_priors(train_tasks, self.config)

        # Latent direct LR->HR trajectory generator artifacts from paired training tasks.
        try:
            from .v31_direct_generators import build_v31_artifacts
            with timer.step("build_latent_direct_generator", n_tasks=len(train_tasks), variant="c"):
                self.v31_artifacts = build_v31_artifacts(train_tasks, variant="c")
        except Exception as exc:
            status(f"Latent direct generator training skipped after error: {exc}")
            self.v31_artifacts = {}

        # Ecological/time-geographic candidate artifacts used by the balanced Top-10 set.
        try:
            from .ecological_candidates import build_ecological_artifacts
            ecological_by_family = {}
            for family in ["step_selection", "constrained_decoder"]:
                with timer.step("build_ecological_candidate_family", n_tasks=len(train_tasks), family=family):
                    ecological_by_family[family] = build_ecological_artifacts(train_tasks, variant=family)
            self.ecological_artifacts = {
                "family": "pooled_step_selection_and_constrained_decoder",
                "artifacts_by_variant": ecological_by_family,
                "n_train_tasks_available": int(len(train_tasks)),
            }
        except Exception as exc:
            status(f"Ecological candidate artifact training skipped after error: {exc}")
            self.ecological_artifacts = {}

        self.task_table = task_table.copy() if task_table is not None else None
        self.metadata = {
            "model_type": "motif_endpoint_reconstruction",
            "n_train_tasks": len(train_tasks),
            "environment_raster_paths": getattr(self, "environment_raster_paths", {}),
            "environment_raster_paths_by_dataset": getattr(self, "environment_raster_paths_by_dataset", {}),
            "environment_epsg_by_dataset": getattr(self, "environment_epsg_by_dataset", {}),
            "n_motifs": int(len(self.library.motif_table)),
            "feature_columns": list(self.library.feature_columns),
            "config": self.config.to_dict(),
            "latent_direct_generator_summary": {
                "variant": self.v31_artifacts.get("variant") if isinstance(self.v31_artifacts, dict) else None,
                "variant_name": self.v31_artifacts.get("variant_name") if isinstance(self.v31_artifacts, dict) else None,
                "n_train_tasks_available": self.v31_artifacts.get("n_train_tasks_available") if isinstance(self.v31_artifacts, dict) else None,
                "n_models_by_n": len(self.v31_artifacts.get("models_by_n", {})) if isinstance(self.v31_artifacts, dict) else 0,
                "has_global_model": bool(self.v31_artifacts.get("global_model")) if isinstance(self.v31_artifacts, dict) else False,
            },
            "ecological_candidate_summary": {
                "family": self.ecological_artifacts.get("family") if isinstance(self.ecological_artifacts, dict) else None,
                "n_train_tasks_available": self.ecological_artifacts.get("n_train_tasks_available") if isinstance(self.ecological_artifacts, dict) else None,
                "n_families": len(self.ecological_artifacts.get("artifacts_by_variant", {})) if isinstance(self.ecological_artifacts, dict) else 0,
            },
        }
        return self

    def _check_fitted(self) -> None:
        if self.library is None or self.movement_priors is None:
            raise RuntimeError("Model has not been fitted or loaded yet.")

    def reconstruct_task(self, task: ReconstructionTask, keep_scored: bool = False) -> ReconstructionResult:
        """Reconstruct one endpoint-conditioned task with the proposed method.

        The user-facing path is the fixed K20 representative using beta
        ``config.softmax_beta``. Baseline paths are not mixed into this proposed
        reconstruction.
        """
        self._check_fitted()
        from .proposed import proposed_paths_for_task

        t0 = time.perf_counter()
        cand_table, cand_paths, cand_time = generate_candidates_for_task(task, self.library, self.config)
        scored, score_time = score_candidates(task, cand_table, cand_paths, self.movement_priors, self.config)
        proposed_paths = proposed_paths_for_task(task, scored, cand_paths, self.config, betas=(float(self.config.softmax_beta),))
        try:
            from .v31_direct_generators import generate_v31_paths_for_task
            proposed_paths.update(generate_v31_paths_for_task(self, task))
        except Exception as exc:
            status(f"Latent direct-generation paths skipped during reconstruction after error: {exc}")
        method = f"pretrained_motif_K20_b{float(self.config.softmax_beta):g}"
        if method not in proposed_paths and proposed_paths:
            method = sorted([m for m in proposed_paths if "K20" in m] or list(proposed_paths))[0]
        if not proposed_paths:
            raise ValueError(f"No proposed paths generated for task {task.task_uid}")
        path, meta = proposed_paths[method]
        # Also keep a diagnostic top1 path when available.
        selected_paths = {"pretrained_motif": path}
        if "pretrained_motif_top1" in proposed_paths:
            selected_paths["pretrained_motif_top1"] = proposed_paths["pretrained_motif_top1"][0]
        total_seconds = time.perf_counter() - t0
        runtime = pd.DataFrame([{**cand_time, **score_time, "task_uid": task.task_uid, "total_seconds": total_seconds}])
        path_table = paths_to_table(task, selected_paths)
        top = pd.DataFrame([{ "task_uid": task.task_uid, "method": "pretrained_motif", **meta }])
        summary = pd.DataFrame([{
            "task_uid": task.task_uid,
            "n_candidates": int(len(scored)),
            "top_candidate_id": meta.get("candidate_id"),
            "top_origin": meta.get("candidate_origin"),
            "top_total_cost": float(scored["proposal_cost"].min()) if "proposal_cost" in scored.columns and len(scored) else np.nan,
            "representative_source_k": int(meta.get("n_source_candidates", np.nan)) if pd.notna(meta.get("n_source_candidates", np.nan)) else np.nan,
            "proposed_method": method,
        }])
        return ReconstructionResult(
            selected_table=top,
            path_table=path_table,
            task_summary=summary,
            runtime_table=runtime,
            scored_candidates=scored if keep_scored else None,
        )

    def reconstruct_tasks(self, tasks: Sequence[ReconstructionTask], keep_scored: bool = False) -> ReconstructionResult:
        """Reconstruct a list of tasks and concatenate output tables."""
        selected_parts = []
        path_parts = []
        summary_parts = []
        runtime_parts = []
        scored_parts = []
        progress = ProgressPrinter("reconstruct_tasks", total=len(tasks), every=max(1, min(25, max(1, len(tasks)//10))))
        for i, task in enumerate(tasks, start=1):
            result = self.reconstruct_task(task, keep_scored=keep_scored)
            selected_parts.append(result.selected_table)
            path_parts.append(result.path_table)
            summary_parts.append(result.task_summary)
            runtime_parts.append(result.runtime_table)
            if keep_scored and result.scored_candidates is not None:
                scored_parts.append(result.scored_candidates)
            progress.update(i, extra=f"latest={task.dataset}/{task.taxon}/{task.setting_name}")
        return ReconstructionResult(
            selected_table=pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(),
            path_table=pd.concat(path_parts, ignore_index=True) if path_parts else pd.DataFrame(),
            task_summary=pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame(),
            runtime_table=pd.concat(runtime_parts, ignore_index=True) if runtime_parts else pd.DataFrame(),
            scored_candidates=pd.concat(scored_parts, ignore_index=True) if scored_parts else None,
        )

    def reconstruct_track(self, coarse_track: pd.DataFrame, fine_interval_min: float, dataset: str = "user", taxon: str = "unknown", animal_id: str = "animal", keep_scored: bool = False) -> ReconstructionResult:
        """Reconstruct a full sparse/coarse trajectory from consecutive gaps."""
        tasks = make_prediction_tasks(coarse_track, fine_interval_min, dataset=dataset, taxon=taxon, animal_id=animal_id)
        if not tasks:
            raise ValueError("No valid consecutive coarse gaps were found.")
        return self.reconstruct_tasks(tasks, keep_scored=keep_scored)

    def save(self, output_dir: str | Path) -> Path:
        """Save the fitted model and human-readable metadata."""
        self._check_fitted()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "pretrained_model.joblib"
        joblib.dump(self, model_path, compress=3)
        write_json(self.metadata, output_dir / "model_metadata.json")
        if self.library is not None:
            write_table(self.library.motif_table, output_dir / "motif_library.parquet")
            write_table(self.library.motif_points, output_dir / "motif_points.parquet")
        if self.movement_priors is not None:
            write_table(self.movement_priors, output_dir / "movement_priors.parquet")
        if isinstance(getattr(self, "v31_artifacts", None), dict) and self.v31_artifacts:
            write_json({
                "variant": self.v31_artifacts.get("variant"),
                "variant_name": self.v31_artifacts.get("variant_name"),
                "n_train_tasks_available": self.v31_artifacts.get("n_train_tasks_available"),
                "n_models_by_n": len(self.v31_artifacts.get("models_by_n", {})),
                "has_global_model": bool(self.v31_artifacts.get("global_model")),
            }, output_dir / "latent_direct_generator_summary.json")
        if isinstance(getattr(self, "ecological_artifacts", None), dict) and self.ecological_artifacts:
            write_json({
                "family": self.ecological_artifacts.get("family"),
                "n_train_tasks_available": self.ecological_artifacts.get("n_train_tasks_available"),
                "n_families": len(self.ecological_artifacts.get("artifacts_by_variant", {})),
            }, output_dir / "ecological_candidate_summary.json")
        return model_path

    @staticmethod
    def load(path: str | Path) -> "MotifReconstructionModel":
        """Load a saved model from ``pretrained_model.joblib``.

        Models saved on Windows may contain pathlib.WindowsPath objects.
        Linux cannot instantiate WindowsPath during unpickling, so map it to
        PosixPath before loading.
        """
        import pathlib
        if not hasattr(pathlib, "_original_windows_path"):
            pathlib._original_windows_path = pathlib.WindowsPath
        pathlib.WindowsPath = pathlib.PosixPath
        obj = joblib.load(path)
        if not isinstance(obj, MotifReconstructionModel):
            raise TypeError(f"Expected MotifReconstructionModel, got {type(obj)!r}")
        if not hasattr(obj, "ecological_artifacts"):
            obj.ecological_artifacts = getattr(obj, "v34_artifacts", {})
        if not hasattr(obj, "v31_artifacts"):
            obj.v31_artifacts = {}
        return obj


def paths_to_table(task: ReconstructionTask, paths: dict[str, np.ndarray]) -> pd.DataFrame:
    """Convert selected path arrays to a long table."""
    rows = []
    for path_id, xy in paths.items():
        for point_order, (x, y) in enumerate(xy):
            rows.append({
                "task_uid": task.task_uid,
                "path_id": path_id,
                "point_order": point_order,
                "time": task.start_time + pd.to_timedelta(point_order * task.fine_dt_min, unit="min"),
                "x": float(x),
                "y": float(y),
            })
    return pd.DataFrame(rows)
