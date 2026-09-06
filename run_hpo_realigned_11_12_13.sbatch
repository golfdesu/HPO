#!/bin/bash
# ==============================================================================
# SLURM Job Script: Fast Runner for Linear & S-Mamba (Models 11, 12, 13)
# Models Covered:
#   - Model 11: DLinear (Zeng et al., AAAI 2023) [Multivariate C=28, Replicate Padding]
#   - Model 12: NLinear (Zeng et al., AAAI 2023) [Multivariate C=28, Channel Independence]
#   - Model 13: S-Mamba (Wang et al., 2024) [Variate Scan M=28, RevIN, 3.5x Speedup]
#
# Estimated Runtime: ~30-45 minutes total for all 3 models (50 trials each)
# ==============================================================================
#SBATCH --job-name=hpo_fast_11_12_13
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --nodelist=compute4
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

echo "Python Executable   : $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
nvidia-smi

mkdir -p logs
mkdir -p best_params

# 2. Sequential Execution of Fast Models (11 -> 12 -> 13)
# ------------------------------------------------------------------------------
SCRIPTS=(
    "11_hpo_dlinear_pytorch.py"
    "12_hpo_nlinear_pytorch.py"
    "13_hpo_smamba_pytorch.py"
)

TOTAL_START=$(date +%s)

for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "================================================================="
    echo ">>> Starting $script at $(date)"
    echo "================================================================="

    SCRIPT_START=$(date +%s)
    python -u "$script"
    EXIT_CODE=$?

    SCRIPT_END=$(date +%s)
    ELAPSED=$((SCRIPT_END - SCRIPT_START))

    echo ""
    echo ">>> Finished $script with exit code $EXIT_CODE (Duration: ${ELAPSED}s / $((ELAPSED / 60))m)"

    json_out="${script%.py}_best_params.json"
    if [ -f "$json_out" ]; then
        cp -v "$json_out" best_params/
    fi

    if [ $EXIT_CODE -ne 0 ]; then
        echo "⚠️ WARNING: $script failed with exit code $EXIT_CODE"
    fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "================================================================="
echo "Fast HPO Jobs (11, 12, 13) Completed at $(date)"
echo "Total Elapsed Time: ${TOTAL_ELAPSED} seconds ($((TOTAL_ELAPSED / 60)) minutes)"
echo "================================================================="
