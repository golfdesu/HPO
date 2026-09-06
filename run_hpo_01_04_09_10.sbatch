#!/bin/bash
# ==============================================================================
# SLURM Job Script: Clean Canonical HPO Runner for Models 01, 04, 09, 10 on Erawan HPC
# Models:
#   - Model 01: Vanilla Transformer Encoder (Vaswani et al., 2017) [GaussianNoise purged]
#   - Model 04: Informer (Zhou et al., AAAI 2021) [Causal Masked Decoder]
#   - Model 09: LSTM Baseline (Hochreiter & Schmidhuber, 1997) [GaussianNoise purged]
#   - Model 10: GRU Baseline (Cho et al., EMNLP 2014) [GaussianNoise purged]
#
# Hardware & Invariants:
#   - Partition: gpu-h100,mixed (1x GPU, 8 CPUs, 32GB RAM)
#   - Eager Mode: PyTorch CUDA Eager with TF32 enabled (NO torch.compile)
#   - Virtual Environment: ../environment/.venv or local .venv
# ==============================================================================
#SBATCH --job-name=hpo_01_04_09_10
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

mkdir -p logs
mkdir -p best_params

# 2. Sequential Execution (01 -> 04 -> 09 -> 10)
# ------------------------------------------------------------------------------
SCRIPTS=(
    "01_hpo_tfm_pytorch.py"
    "04_hpo_ifm_pytorch.py"
    "09_hpo_lstm_pytorch.py"
    "10_hpo_gru_pytorch.py"
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

    # Copy output JSON to best_params directory
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
echo "All Clean HPO Jobs (01, 04, 09, 10) Completed at $(date)"
echo "Total Elapsed Time: ${TOTAL_ELAPSED} seconds ($((TOTAL_ELAPSED / 60)) minutes)"
echo "================================================================="
