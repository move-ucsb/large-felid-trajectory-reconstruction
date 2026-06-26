# Large-Felid Trajectory Reconstruction

Pretrained motif-retrieval and probabilistic time-geographic reconstruction of sparse large-felid GPS trajectories.

The package provides:

- **zero-shot reconstruction** using the included pretrained model;
- **few-shot adaptation** using a small amount of high-resolution data from a new animal/study area; and
- **full retraining** from user-provided high-resolution tracking data.

## Repository layout

```text
src/carnivore_reconstruction/      Core Python code
config/dataset_specs.py            Edit this when training your own model
notebooks/                         Numbered workflow notebooks
scripts/                           HPC/check scripts
outputs/pretrained_model/          Included pretrained model and task tables
data/raw/                          Empty folder for user high-resolution tracks
data/raster/                       Empty folder for optional raster covariates
```

The included pretrained model was trained for large-felid trajectory reconstruction. The final paper-facing methods are:

```text
robust_pretrained_top1
balanced_pooled_top10_candidate_set
```

## Install

Create an environment and install the package:

```bash
python -m venv ~/venvs/geo_env
source ~/venvs/geo_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ipykernel install --user --name geo_env --display-name "Python (geo_env)"
```

If your shell is `/usr/bin/sh`, use `.` instead of `source`:

```bash
. ~/venvs/geo_env/bin/activate
```

Check the environment:

```bash
python -c "import pandas, pyarrow, sklearn, joblib; print('environment OK')"
```

## Zero-shot use: reconstruct a new sparse trajectory

Use this when you have only a coarse/sparse trajectory and want to reconstruct plausible finer-scale paths using the included pretrained model.

Input CSV should contain:

```text
time,x,y
```

where `time` is a timestamp and `x,y` are projected coordinates in meters.

Example:

```python
from pathlib import Path
import pandas as pd
from carnivore_reconstruction import MotifReconstructionModel

PROJECT_ROOT = Path.cwd()
model = MotifReconstructionModel.load(
    PROJECT_ROOT / "outputs" / "pretrained_model" / "pretrained_model.joblib"
)

coarse = pd.read_csv("my_sparse_track.csv")
coarse["time"] = pd.to_datetime(coarse["time"])

result = model.reconstruct_track(
    coarse_track=coarse,
    fine_interval_min=5,      # desired output interval, e.g. 5 minutes
    dataset="my_study_area",
    taxon="puma",             # use closest known group: puma/cougar/tiger/leopard
    animal_id="animal_001",
    keep_scored=True,
)

result.selected_table.to_csv("zero_shot_selected_paths.csv", index=False)
result.path_table.to_csv("zero_shot_path_points.csv", index=False)
result.task_summary.to_csv("zero_shot_task_summary.csv", index=False)

if result.scored_candidates is not None:
    result.scored_candidates.to_csv("zero_shot_scored_candidates.csv", index=False)
```

`selected_table` gives the selected reconstruction for each coarse gap.  
`path_table` gives reconstructed path points.  
`scored_candidates` is useful when comparing candidate alternatives.

## Few-shot use: adapt with high-resolution examples

Use this when you have some high-resolution trajectories from the same species, individual, or study area. Few-shot adaptation adds calibration motifs and then reuses the same reconstruction pipeline.

```python
import pandas as pd
from carnivore_reconstruction import (
    MotifReconstructionModel,
    ReconstructionConfig,
    make_tasks_from_tracks,
)
from carnivore_reconstruction.fewshot import adapt_model_with_calibration

base_model = MotifReconstructionModel.load(
    "outputs/pretrained_model/pretrained_model.joblib"
)

# High-resolution calibration data should contain:
# animal_id, time, x, y, dataset, taxon
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

Few-shot adaptation is most useful when the calibration examples come from the same animal, population, habitat, or sampling design as the target sparse trajectory.

## Train your own model

Use this when you have enough high-resolution trajectories and want to build a new pretrained model.

### 1. Put data in `data/raw/`

Expected high-resolution CSV columns:

```text
animal_id,time,x,y
```

or longitude/latitude columns if you configure `lon_col`, `lat_col`, and `epsg`.

### 2. Optional rasters

Put raster covariates in:

```text
data/raster/
```

Rasters are optional. They are not needed for zero-shot reconstruction or the default validation/test workflow.

### 3. Edit `config/dataset_specs.py`

Example projected-coordinate dataset:

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

Example lon/lat dataset:

```python
DatasetSpec(
    path=DATA_ROOT / "your_lonlat_tracks.csv",
    dataset="Your_lonlat_dataset",
    taxon="cougar",
    id_col="animal_id",
    time_col="time",
    x_col=None,
    y_col=None,
    lon_col="longitude",
    lat_col="latitude",
    epsg=32610,
)
```

### 4. Run notebook 0

```text
notebooks/0_build_pretrained_model.ipynb
```

This writes a new model to:

```text
outputs/pretrained_model/
```

## Validation and test workflow

Run notebooks in order:

```text
0_build_pretrained_model.ipynb        # optional if using included pretrained model
1_formal_validation.ipynb
2_formal_heldout_test.ipynb
3_leave_one_individual_out.ipynb
4_transfer_analysis.ipynb
5_make_publication_figures.ipynb
```

If using the included pretrained model, start from notebook 1.

## HPC use

Check the pretrained files:

```bash
python scripts/hpc_check_pretrained.py
```

Run validation/test/figures:

```bash
bash scripts/hpc_run_validation_test.sh
```

or with SLURM:

```bash
sbatch scripts/slurm_validation_test.sh
```

Run LOIO separately:

```bash
bash scripts/hpc_run_loio.sh
```

Recommended resources:

```text
1 node
16 CPUs
64 GB RAM
24–72 hours
no GPU
```

## Important notes

- The included pretrained folder is large. For a public GitHub repository, Git LFS or a GitHub Release asset is usually cleaner than committing model binaries directly.
- Do not commit private/raw tracking data.
- Keep user-provided raw files in `data/raw/`.
- Keep optional raster files in `data/raster/`.
- Generated validation/test outputs are written under `outputs/`.

## Citation

If you use this package, cite the associated dissertation chapter/manuscript when available.

```text
Liu Y., Dodge S., Wilmers C.C. A Context-Sensitive Probabilistic Time-Geographic Approach to Interpolating Sparse Carnivore Trajectories. Manuscript in preparation
```
