#!/bin/bash
# ==============================================================================
# SLURM Job Script: Model 00 (Custom Transformer 1D HPO) on NVIDIA H100
# Cluster: Erawan HPC (NSTDA / CMU)
# Partition: gpu-h100 (NVIDIA H100 80GB HBM3)
# Optimizes: Attention Orthogonal Regularization Strength (50 trials, 30 epochs)
# ==============================================================================
#SBATCH --job-name=hpo_00_tfm
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Running on Node     : $SLURMD_NODENAME"
echo "Assigned Partition  : $SLURM_JOB_PARTITION"
echo "Start Time          : $(date)"
echo "Current Directory   : $(pwd)"
echo "================================================================="

# 1. Environment & Module Setup
if command -v module &> /dev/null; then
    module purge
    module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi

# Activate Virtual Environment
if [ -d "$HOME/project/HPO/.venv" ]; then
    source "$HOME/project/HPO/.venv/bin/activate"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$HOME/.venv" ]; then
    source "$HOME/.venv/bin/activate"
fi

mkdir -p logs

# 2. Hardware Diagnostics
echo "Python Executable   : $(which python)"
python -c "import torch; print(f'PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
nvidia-smi

# 3. Launch Model 00 Optuna HPO
echo ""
echo ">>> Launching Model 00 (Custom Transformer 1D HPO - 50 Trials)..."
START_TIME=$(date +%s)
python -u 00_hpo_tfm_custom_pytorch.py
EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "================================================================="
echo "Finished 00_hpo_tfm_custom_pytorch.py with exit code $EXIT_CODE (Elapsed: ${ELAPSED}s) at $(date)"
echo "================================================================="
exit $EXIT_CODE
