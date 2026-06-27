# Large-Felid Trajectory Reconstruction

Pretrained motif-retrieval and probabilistic time-geographic reconstruction for sparse large-felid GPS trajectories.

The package supports:

- **Zero-shot reconstruction** with the included pretrained model.
- **Few-shot adaptation** with limited high-resolution calibration data.
- **Full retraining** from user-provided high-resolution tracks.

## Repository layout

```text
src/carnivore_reconstruction/   Core Python package
config/dataset_specs.py         Dataset configuration for training
notebooks/                      Workflow notebooks
scripts/                        HPC and utility scripts
outputs/pretrained_model/       Included pretrained model and task tables
data/raw/                       User tracking data
data/raster/                    Optional raster covariates
```

The main paper-facing methods are:

```text
robust_pretrained_top1
balanced_pooled_top10_candidate_set
```

## Installation

```bash
python -m venv ~/venvs/geo_env
source ~/venvs/geo_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ipykernel install --user --name geo_env --display-name "Python (geo_env)"
```

Check the environment:

```bash
python -c "import pandas, pyarrow, sklearn, joblib; print('environment OK')"
```

## Zero-shot reconstruction

Use this when you only have a sparse trajectory. Input coordinates should be projected in meters.

Required CSV columns:

```text
time,x,y
```

Example:

```python
from pathlib import Path
import pandas as pd
from carnivore_reconstruction import MotifReconstructionModel

model = MotifReconstructionModel.load(
    Path("outputs/pretrained_model/pretrained_model.joblib")
)

coarse = pd.read_csv("my_sparse_track.csv")
coarse["time"] = pd.to_datetime(coarse["time"])

result = model.reconstruct_track(
    coarse_track=coarse,
    fine_interval_min=5,
    dataset="my_study_area",
    taxon="puma",        # closest known group: puma/cougar/tiger/leopard
    animal_id="animal_001",
    keep_scored=True,
)

result.selected_table.to_csv("zero_shot_selected_paths.csv", index=False)
result.path_table.to_csv("zero_shot_path_points.csv", index=False)
result.task_summary.to_csv("zero_shot_task_summary.csv", index=False)

if result.scored_candidates is not None:
    result.scored_candidates.to_csv("zero_shot_scored_candidates.csv", index=False)
```

## Few-shot adaptation

Use this when some high-resolution calibration data are available from the target animal, population, habitat, or sampling design.

```python
import pandas as pd
from carnivore_reconstruction import MotifReconstructionModel, ReconstructionConfig, make_tasks_from_tracks
from carnivore_reconstruction.fewshot import adapt_model_with_calibration

base_model = MotifReconstructionModel.load("outputs/pretrained_model/pretrained_model.joblib")

cal_tracks = pd.read_csv("my_high_resolution_calibration_tracks.csv")
cal_tracks["time"] = pd.to_datetime(cal_tracks["time"])

config = ReconstructionConfig(
    coarse_intervals_min=(60, 120, 240),
    fine_interval_min=5,
    output_dir="outputs/fewshot_model",
    evaluate_environmental_exposure=False,
    save_candidate_paths=False,
)

task_table, task_points, calibration_tasks = make_tasks_from_tracks(cal_tracks, config)

fewshot_model = adapt_model_with_calibration(
    base_model,
    calibration_tasks,
    max_calibration_motifs=5000,
)

coarse = pd.read_csv("my_sparse_track.csv")
coarse["time"] = pd.to_datetime(coarse["time"])

result = fewshot_model.reconstruct_track(
    coarse,
    fine_interval_min=5,
    dataset="my_study_area",
    taxon="puma",
    animal_id="animal_001",
    keep_scored=True,
)

result.path_table.to_csv("fewshot_path_points.csv", index=False)
```

## Training a new model

1. Put high-resolution tracks in `data/raw/`.

Required columns:

```text
animal_id,time,x,y
```

Longitude/latitude inputs are also supported if `lon_col`, `lat_col`, and `epsg` are set in `config/dataset_specs.py`.

2. Optional raster covariates can be placed in `data/raster/`.

3. Edit `config/dataset_specs.py`.

Example:

```python
DatasetSpec(
    path=DATA_ROOT / "your_high_resolution_tracks.csv",
    dataset="Your_dataset_name",
    taxon="puma",
    id_col="animal_id",
    time_col="time",
    x_col="x",
    y_col="y",
)
```

4. Run:

```text
notebooks/0_build_pretrained_model.ipynb
```

The new model will be written to `outputs/pretrained_model/`.

## Evaluation workflow

If using the included pretrained model, start from validation:

```text
1_formal_validation.ipynb
2_formal_heldout_test.ipynb
3_leave_one_individual_out.ipynb
4_transfer_analysis.ipynb
5_make_publication_figures.ipynb
```

## HPC workflow

Check pretrained files:

```bash
python scripts/hpc_check_pretrained.py
```

Run validation, held-out test, transfer analysis, and figures:

```bash
bash scripts/hpc_run_validation_test.sh
```

Run LOIO separately:

```bash
bash scripts/hpc_run_loio.sh
```

Suggested resources:

```text
1 node
16 CPUs
64 GB RAM
24–72 hours
no GPU
```

## Notes

- Do not commit private or raw tracking data.
- Use Git LFS or a GitHub Release asset for large pretrained model files.
- Rasters are optional for zero-shot use and default validation/test runs.
- Generated outputs are written under `outputs/`.

## Citation

If you use this package, cite the associated dissertation chapter or manuscript when available:

```text
Liu Y., Dodge S., Wilmers C.C. A Context-Sensitive Probabilistic Time-Geographic Approach to Interpolating Sparse Carnivore Trajectories. Manuscript in preparation.
```
