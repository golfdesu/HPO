#!/bin/bash
# ==============================================================================
# SLURM Job Script: Sequential Runner for Files 17-20 on Erawan HPC (ThaiSC)
# Models: 17 (S-Mamba), 18 (PowerMamba), 19 (TimeMachine), 20 (S4D)
# ==============================================================================
#SBATCH --job-name=hpo_17_20
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu-h100,mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

echo "================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Running on Node     : $SLURMD_NODENAME"
echo "Start Time          : $(date)"
echo "Current Directory   : $(pwd)"
echo "================================================================="

# 1. Setup Environment & Modules
# ------------------------------------------------------------------------------
# Auto-load cuda/compiler modules if available on Erawan (OpenHPC/Lmod)
if command -v module &> /dev/null; then
    module purge
    module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi

# Activate Virtual Environment if present
if [ -d "$HOME/project/HPO/.venv" ]; then
    source "$HOME/project/HPO/.venv/bin/activate"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$HOME/.venv" ]; then
    source "$HOME/.venv/bin/activate"
fi

# Print Python and GPU Diagnostics
echo "Python Path         : $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
nvidia-smi

mkdir -p logs

# 2. Sequential Execution (17 -> 18 -> 19 -> 20)
# ------------------------------------------------------------------------------
SCRIPTS=(
    "13_hpo_smamba_pytorch.py"
    "14_hpo_powermamba_pytorch.py"
    "15_hpo_timemachine_pytorch.py"
    "16_hpo_s4d_pytorch.py"
)

for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "================================================================="
    echo ">>> Starting $script at $(date)"
    echo "================================================================="

    # -u flag for unbuffered stdout/stderr so logs flush to file in real-time
    python -u "$script"
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] $script exited with code $EXIT_CODE at $(date)"
    else
        echo "[SUCCESS] $script completed successfully at $(date)"
    fi
done

echo ""
echo "================================================================="
echo "All tasks finished at $(date)"
echo "================================================================="
