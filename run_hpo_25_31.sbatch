#!/bin/bash
# ==============================================================================
# SLURM Job Script: Sequential HPO Runner for Models 25-31 on Erawan HPC (ThaiSC)
# Target Architecture: NVIDIA H100 80GB HBM3 (Hopper)
# Models:
#   - Model 25: TiDE (Time-series Dense Encoder - Google Research, TMLR 2023)
#   - Model 26: N-BEATS (Neural Basis Expansion Analysis - ServiceNow, ICLR 2020)
#   - Model 27: ModernTCN (Pure Convolution Structure - ICLR 2024 Spotlight)
#   - Model 28: Crossformer (Cross-Scale Dimension Transformer - ICLR 2023)
#   - Model 29: SegRNN (Segment Recurrent Neural Network - ICLR 2024)
#   - Model 30: Non-stationary Transformer (De-stationary Attention - NeurIPS 2022)
#   - Model 31: SCINet (Sample Convolution and Interaction - NeurIPS 2022)
#
# Hardware & Invariants:
#   - Partition: gpu-h100,mixed (1x NVIDIA H100 80GB, 8 CPUs, 32GB RAM)
#   - Eager Mode: Pure PyTorch CUDA Eager with TF32 enabled (NO torch.compile)
#   - Time Limit: 24:00:00 (24 hours)
#   - Virtual Environment: ../environment/.venv or .venv
# ==============================================================================
#SBATCH --job-name=hpo_25_31
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

# Print Python and GPU Diagnostics
echo "Python Executable   : $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
nvidia-smi

mkdir -p logs best_params configs outputs

# 2. Sequential Execution (25 -> 26 -> 27 -> 28 -> 29 -> 30 -> 31)
# ------------------------------------------------------------------------------
SCRIPTS=(
    "25_hpo_tide_pytorch.py"
    "26_hpo_nbeats_pytorch.py"
    "27_hpo_moderntcn_pytorch.py"
    "28_hpo_crossformer_pytorch.py"
    "29_hpo_segrnn_pytorch.py"
    "30_hpo_nstransformer_pytorch.py"
    "31_hpo_scinet_pytorch.py"
)

TOTAL_START=$(date +%s)
FAILED_MODELS=()

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
    echo ">>> Finished $script with exit code $EXIT_CODE (Duration: ${ELAPSED}s)"

    if [ $EXIT_CODE -ne 0 ]; then
        echo "⚠️ WARNING: $script failed with exit code $EXIT_CODE"
        FAILED_MODELS+=("$script (Code: $EXIT_CODE)")
    else
        echo "✅ SUCCESS: $script completed successfully."
    fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "================================================================="
echo "All HPO Jobs (Models 25-31) Completed"
echo "End Time            : $(date)"
echo "Total Wall Time     : $((TOTAL_ELAPSED / 3600))h $(((TOTAL_ELAPSED % 3600) / 60))m $((TOTAL_ELAPSED % 60))s"

if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
    echo "Status              : ALL 7 MODELS FINISHED SUCCESSFULLY ✅"
else
    echo "Status              : SOME MODELS ENCOUNTERED ISSUES ⚠️"
    echo "Failed List         : ${FAILED_MODELS[*]}"
fi
echo "================================================================="
