#!/usr/bin/env bash
#SBATCH --job-name=pepp-mace-val
#SBATCH --output=logs/slurm_validate_%j.out
#SBATCH --error=logs/slurm_validate_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=TODO_PARTITION
#SBATCH --account=TODO_ACCOUNT

set -euo pipefail

# TODO: module purge
# TODO: module load CUDA_MODULE
# TODO: source /path/to/conda.sh && conda activate /path/to/env
# TODO: or source /path/to/venv/bin/activate

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
python3 scripts/04_validate_model.py
