#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Project: $(pwd)"

export CARNIVORE_N_JOBS="${CARNIVORE_N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"
if [ "$CARNIVORE_N_JOBS" -gt 2 ]; then
  export CARNIVORE_N_JOBS=$((CARNIVORE_N_JOBS-2))
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CARNIVORE_FAST_EVAL=1

echo "CARNIVORE_N_JOBS=$CARNIVORE_N_JOBS"

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

python -m jupyter nbconvert --to notebook --execute notebooks/3_leave_one_individual_out.ipynb \
  --output 3_leave_one_individual_out_executed.ipynb \
  --output-dir outputs/executed_notebooks \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=geo_env 2>&1 | tee logs/07_leave_one_individual_out.log

echo "Done: $(date)"
df -h .
