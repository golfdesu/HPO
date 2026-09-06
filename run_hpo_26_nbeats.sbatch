#!/bin/bash
# ==============================================================================
# SLURM Job Script: Dedicated HPO Runner for Model 26 (N-BEATS) on Erawan HPC
# Target Architecture: NVIDIA H100 80GB HBM3 (Hopper)
# ==============================================================================
#SBATCH --job-name=hpo_26_nbeats
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu-h100,mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Running on Node     : $SLURMD_NODENAME"
echo "Start Time          : $(date)"
echo "Current Directory   : $(pwd)"
echo "================================================================="

# 1. Setup Environment & Modules
if command -v module &> /dev/null; then
    module purge
    module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi

# Activate Virtual Environment
if [ -f "../environment/.venv/bin/activate" ]; then
    echo "Activating virtualenv at ../environment/.venv..."
    source ../environment/.venv/bin/activate
elif [ -d "../environment/.venv" ]; then
    echo "Activating virtualenv at ../environment/.venv..."
    source ../environment/.venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "Activating local virtualenv at .venv..."
    source .venv/bin/activate
elif [ -d "$HOME/project/HPO/.venv" ]; then
    echo "Activating virtualenv at $HOME/project/HPO/.venv..."
    source "$HOME/project/HPO/.venv/bin/activate"
else
    echo "Warning: No virtualenv found, using default python: $(which python)"
fi

mkdir -p logs best_params configs outputs

echo ""
echo "================================================================="
echo ">>> Starting 26_hpo_nbeats_pytorch.py at $(date)"
echo "================================================================="

python -u 26_hpo_nbeats_pytorch.py
EXIT_CODE=$?

echo ""
echo ">>> Finished 26_hpo_nbeats_pytorch.py with exit code $EXIT_CODE"
echo "================================================================="
