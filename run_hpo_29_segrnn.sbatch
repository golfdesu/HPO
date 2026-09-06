#!/bin/bash
# ==============================================================================
# SLURM Job Script: Model 29 SegRNN HPO Runner on 'mixed' Partition
# Model: SegRNN (Lin et al., ICLR 2024) [Segment GRU + PMF Decoding]
# Hardware Target: 'mixed' partition
# ==============================================================================
#SBATCH --job-name=hpo_29_segrnn
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Running on Node     : $SLURMD_NODENAME"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Assigned Partition  : $SLURM_JOB_PARTITION"
echo "Start Time          : $(date)"
echo "Current Directory   : $(pwd)"
echo "================================================================="

# 1. Environment & Module Setup
# ------------------------------------------------------------------------------
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

# Print Diagnostics
echo "Python Executable   : $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available or no dedicated GPU allocated"

mkdir -p logs
mkdir -p best_params

# 2. Execution of Model 29
# ------------------------------------------------------------------------------
echo ""
echo "================================================================="
echo ">>> Starting 29_hpo_segrnn_pytorch.py at $(date)"
echo "================================================================="

SCRIPT_START=$(date +%s)
python -u 29_hpo_segrnn_pytorch.py
EXIT_CODE=$?
SCRIPT_END=$(date +%s)
ELAPSED=$((SCRIPT_END - SCRIPT_START))

if [ $EXIT_CODE -eq 0 ]; then
    echo ">>> [SUCCESS] 29_hpo_segrnn_pytorch.py completed in ${ELAPSED}s at $(date)"
else
    echo ">>> [ERROR] 29_hpo_segrnn_pytorch.py FAILED with exit code $EXIT_CODE after ${ELAPSED}s at $(date)"
fi

echo ""
echo "================================================================="
echo "Job Finished at     : $(date)"
echo "Total Walltime      : ${ELAPSED}s ($((ELAPSED / 60)) mins)"
echo "Output Directory    : best_params/"
echo "================================================================="
