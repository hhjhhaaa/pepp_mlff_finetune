#!/usr/bin/env bash
#SBATCH --job-name=pepp-mace-ft
#SBATCH --output=logs/slurm_finetune_%j.out
#SBATCH --error=logs/slurm_finetune_%j.err
#SBATCH --time=24:00:00
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
test ! -e runs/pepp_mace_off_finetune_v001 || echo "Run directory exists; MACE config controls overwrite behavior."
bash scripts/03_finetune_mace.sh --run
