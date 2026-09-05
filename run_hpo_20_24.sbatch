#!/bin/bash
# ==============================================================================
# SLURM Job Script: Sequential HPO Runner for Models 20-24 on Erawan HPC (ThaiSC)
# Models:
#   - Model 20: Multi-scale Fusion Transformer (MFT)
#   - Model 21: CNN-LSTM-Transformer Hybrid
#   - Model 22: FEDformer (Frequency Enhanced Decomposed Transformer)
#   - Model 23: Temporal Convolutional Network (TCN)
#   - Model 24: N-HiTS (Neural Hierarchical Interpolation)
#
# Hardware & Invariants:
#   - Partition: mixed,gpu-h100 (1x GPU, 8 CPUs, 32GB RAM)
#   - Eager Mode: PyTorch CUDA Eager with TF32 enabled
#   - Virtual Environment: ../environment/.venv
# ==============================================================================
#SBATCH --job-name=hpo_20_24
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

# Activate Virtual Environment at ../environment/.venv
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
    echo "Warning: No virtualenv found at ../environment/.venv, using default python: $(which python)"
fi

# Print Python and GPU Diagnostics
echo "Python Executable   : $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
nvidia-smi

mkdir -p logs

# 2. Sequential Execution (20 -> 21 -> 22 -> 23 -> 24)
# ------------------------------------------------------------------------------
SCRIPTS=(
    "20_hpo_mft_pytorch.py"
    "21_hpo_cnn_lstm_tfm_pytorch.py"
    "22_hpo_fedformer_pytorch.py"
    "23_hpo_tcn_pytorch.py"
    "24_hpo_nhits_pytorch.py"
)

TOTAL_START=$(date +%s)

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
        # Continue with next model so one failure doesn't block the entire batch
    fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "================================================================="
echo "All HPO Jobs (20-24) Completed at $(date)"
echo "Total Elapsed Time: ${TOTAL_ELAPSED} seconds ($((TOTAL_ELAPSED / 60)) minutes)"
echo "================================================================="
