#!/bin/bash
#SBATCH --job-name=felid_val_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_val_test_%j.out
#SBATCH --error=logs/slurm_val_test_%j.err

# Submit from the project root:
#   sbatch scripts/slurm_validation_test.sh

set -euo pipefail
mkdir -p logs
bash scripts/hpc_run_validation_test.sh
