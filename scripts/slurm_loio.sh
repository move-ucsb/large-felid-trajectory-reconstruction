#!/bin/bash
#SBATCH --job-name=felid_loio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_loio_%j.out
#SBATCH --error=logs/slurm_loio_%j.err

# Submit from the project root:
#   sbatch scripts/slurm_loio.sh

set -euo pipefail
mkdir -p logs
bash scripts/hpc_run_loio.sh
