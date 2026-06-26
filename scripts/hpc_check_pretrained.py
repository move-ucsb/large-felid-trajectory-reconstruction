#!/usr/bin/env python
"""Check that the HPC package has the pretrained outputs needed for evaluation."""
from pathlib import Path
import os
import shutil
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "pretrained_model"

def exists_auto(name: str) -> Path | None:
    for suffix in [".parquet", ".csv"]:
        p = MODEL_DIR / f"{name}{suffix}"
        if p.exists():
            return p
    return None

def read_auto(p: Path):
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)

print("Project root:", ROOT)
print("Model dir:", MODEL_DIR)
print("Free space at project root:", shutil.disk_usage(ROOT))
print("CARNIVORE_N_JOBS:", os.environ.get("CARNIVORE_N_JOBS"))
print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"))

required = {
    "pretrained_model.joblib": MODEL_DIR / "pretrained_model.joblib",
    "task_table": exists_auto("task_table"),
    "task_points": exists_auto("task_points"),
}
ok = True
for name, path in required.items():
    if path is None or not Path(path).exists():
        print(f"MISSING: {name}")
        ok = False
    else:
        print(f"FOUND: {name}: {path} ({Path(path).stat().st_size/1e6:.1f} MB)")

split = MODEL_DIR / "formal_individual_split_table.csv"
print("formal_individual_split_table.csv:", "FOUND" if split.exists() else "missing; notebooks can repair/regenerate split labels if needed")

if required["task_table"] is not None and Path(required["task_table"]).exists():
    tt = read_auto(Path(required["task_table"]))
    print("task_table shape:", tt.shape)
    if "split" in tt.columns:
        print("split counts:")
        print(tt["split"].value_counts(dropna=False).to_string())
    cols = [c for c in ["dataset", "taxon", "setting_name", "split"] if c in tt.columns]
    if cols:
        print("task counts by dataset/taxon/setting/split:")
        print(tt.groupby(cols, dropna=False).size().reset_index(name="n_tasks").head(80).to_string(index=False))

if not ok:
    print("\nERROR: pretrained outputs are incomplete. Copy the whole outputs/pretrained_model folder before running validation/test.")
    sys.exit(1)

print("\nPretrained output check passed. Raster files are not required for validation/test/LOIO in this HPC package.")
