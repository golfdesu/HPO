#!/bin/bash
# ==============================================================================
# SLURM Job Script: Transformer Models HPO Runner on 'mixed' Partition (Models 02, 05, 06)
# Target Models:
#   - Model 02: Vanilla Transformer Decoder (Vaswani et al., NeurIPS 2017)
#   - Model 05: Autoformer (Wu et al., NeurIPS 2021) [AutoCorrelation + Series Decomp]
#   - Model 06: PatchTST (Nie et al., ICLR 2023) [Patching + Channel Independence]
#
# Hardware Target: 'mixed' partition
# Estimated Runtime: ~2.0 - 3.0 hours total (50 trials each)
# ==============================================================================
#SBATCH --job-name=hpo_02_05_06
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

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

# Print Diagnostics
echo "Python Executable   : $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
nvidia-smi

mkdir -p logs
mkdir -p best_params

# 2. Sequential Execution (Models 02, 05, 06)
# ------------------------------------------------------------------------------
SCRIPTS=(
    "02_hpo_dec_pytorch.py"
    "05_hpo_afm_pytorch.py"
    "06_hpo_ptst_pytorch.py"
)

TOTAL_START=$(date +%s)
SUCCESS_COUNT=0
FAIL_COUNT=0

for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "================================================================="
    echo ">>> Starting $script at $(date)"
    echo "================================================================="

    SCRIPT_START=$(date +%s)
    
    # -u flag for unbuffered stdout/stderr so logs flush to file in real-time
    python -u "$script"
    EXIT_CODE=$?

    SCRIPT_END=$(date +%s)
    ELAPSED=$((SCRIPT_END - SCRIPT_START))

    if [ $EXIT_CODE -eq 0 ]; then
        echo ">>> [SUCCESS] $script completed in ${ELAPSED}s at $(date)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ">>> [ERROR] $script FAILED with exit code $EXIT_CODE after ${ELAPSED}s at $(date)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "================================================================="
echo "All Jobs Finished at: $(date)"
echo "Total Walltime      : ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60)) mins)"
echo "Success Count       : $SUCCESS_COUNT / ${#SCRIPTS[@]}"
echo "Failure Count       : $FAIL_COUNT / ${#SCRIPTS[@]}"
echo "Output Directory    : best_params/"
echo "================================================================="
