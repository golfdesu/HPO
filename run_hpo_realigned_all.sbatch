#!/bin/bash
# ==============================================================================
# SLURM Job Script: Master Unified HPO Runner on Erawan HPC (8 Models)
# Bundles all updated, purged, and re-aligned models into a single sequential queue:
#
# Fast Tier (< 10 min total):
#   - Model 11: DLinear (Zeng et al., AAAI 2023) [Multivariate C=28, Replicate Padding]
#   - Model 12: NLinear (Zeng et al., AAAI 2023) [Multivariate C=28, Channel Independence]
#   - Model 09: LSTM Baseline (1997) [GaussianNoise purged, clean canonical baseline]
#   - Model 10: GRU Baseline (2014) [GaussianNoise purged, clean canonical baseline]
#
# Deep Tier (~15-30 min each):
#   - Model 13: S-Mamba (Wang et al., 2024) [Variate Scan M=28, RevIN, 3.5x Speedup]
#   - Model 01: Vanilla Transformer Encoder (Vaswani 2017) [Fixed Sinusoidal PE Buffer]
#   - Model 03: Full Seq2Seq Transformer (Vaswani 2017) [Zero Queries + Token Head]
#   - Model 04: Informer (Zhou et al., AAAI 2021) [FullAttention Cross-Attn + Token Head]
#
# Estimated Total Walltime: ~1.5 - 2.5 hours on NVIDIA H100 (Allocated: 24:00:00)
# ==============================================================================
#SBATCH --job-name=hpo_master_8models
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --nodelist=compute4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

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

# 2. Sequential Execution (Ordered strategically from fastest to deepest)
# ------------------------------------------------------------------------------
SCRIPTS=(
    "11_hpo_dlinear_pytorch.py"
    "12_hpo_nlinear_pytorch.py"
    "09_hpo_lstm_pytorch.py"
    "10_hpo_gru_pytorch.py"
    "13_hpo_smamba_pytorch.py"
    "01_hpo_tfm_pytorch.py"
    "03_hpo_encdec_pytorch.py"
    "04_hpo_ifm_pytorch.py"
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

    echo ""
    echo ">>> Finished $script with exit code $EXIT_CODE (Duration: ${ELAPSED}s / $((ELAPSED / 60))m)"

    # Copy output JSON to best_params directory
    json_out="${script%.py}_best_params.json"
    if [ -f "$json_out" ]; then
        cp -v "$json_out" best_params/
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "⚠️ WARNING: $script failed with exit code $EXIT_CODE"
    fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "================================================================="
echo "All 8 HPO Jobs Completed at $(date)"
echo "Success: ${SUCCESS_COUNT}/${#SCRIPTS[@]} | Failed: ${FAIL_COUNT}/${#SCRIPTS[@]}"
echo "Total Elapsed Time: ${TOTAL_ELAPSED} seconds ($((TOTAL_ELAPSED / 60)) minutes)"
echo "================================================================="
