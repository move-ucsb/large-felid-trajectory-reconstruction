#!/bin/bash
set -euo pipefail

# Run inside an allocated HPC session or from a SLURM batch script.
cd "$(dirname "$0")/.."

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Project: $(pwd)"

# Use allocated CPUs, leaving a little room for system/Jupyter overhead.
export CARNIVORE_N_JOBS="${CARNIVORE_N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"
if [ "$CARNIVORE_N_JOBS" -gt 2 ]; then
  export CARNIVORE_N_JOBS=$((CARNIVORE_N_JOBS-2))
fi

# Avoid oversubscription from BLAS/OpenMP inside each task thread.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CARNIVORE_FAST_EVAL=1

echo "CARNIVORE_N_JOBS=$CARNIVORE_N_JOBS"

# Activate environment. Edit this block if your HPC uses a different setup.
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate geo_env || true
fi
if [ -f "$HOME/.conda/envs/geo_env/bin/activate" ]; then
  source "$HOME/.conda/envs/geo_env/bin/activate"
fi
if [ -f "$HOME/venvs/geo_env/bin/activate" ]; then
  source "$HOME/venvs/geo_env/bin/activate"
fi

python --version
python scripts/hpc_check_pretrained.py

mkdir -p outputs/executed_notebooks logs

python -m jupyter nbconvert --to notebook --execute notebooks/1_formal_validation.ipynb \
  --output 1_formal_validation_executed.ipynb \
  --output-dir outputs/executed_notebooks \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=geo_env 2>&1 | tee logs/05_formal_validation.log

python -m jupyter nbconvert --to notebook --execute notebooks/2_formal_heldout_test.ipynb \
  --output 2_formal_heldout_test_executed.ipynb \
  --output-dir outputs/executed_notebooks \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=geo_env 2>&1 | tee logs/06_formal_heldout_test.log

python -m jupyter nbconvert --to notebook --execute notebooks/4_transfer_analysis.ipynb \
  --output 4_transfer_analysis_executed.ipynb \
  --output-dir outputs/executed_notebooks \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=geo_env 2>&1 | tee logs/08_transfer_analysis.log

python -m jupyter nbconvert --to notebook --execute notebooks/5_make_publication_figures.ipynb \
  --output 5_make_publication_figures_executed.ipynb \
  --output-dir outputs/executed_notebooks \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=geo_env 2>&1 | tee logs/09_make_publication_figures.log

echo "Done: $(date)"
df -h .
