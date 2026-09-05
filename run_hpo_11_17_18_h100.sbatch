#!/bin/bash
# ==============================================================================
# SLURM Job Script: Concurrent Runner for Models 11, 17, and 18 on NVIDIA H100
# - Model 11 (LightGBM)   : Runs on AMD EPYC CPU Multi-threading
# - Model 17 (S-Mamba)    : Runs on NVIDIA H100 80GB GPU
# - Model 18 (PowerMamba) : Runs on NVIDIA H100 80GB GPU
# ==============================================================================
#SBATCH --job-name=hpo_11_17_18
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=gpu-h100              # Strictly H100 partition only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                # 16 CPU cores (LightGBM: 8, Mamba: 4 each)
#SBATCH --gres=gpu:1                      # 1x NVIDIA H100 80GB GPU
#SBATCH --mem=32G                         # 32GB RAM (plenty for all 3 models)
#SBATCH --time=12:00:00                   # 12 Hours max walltime

echo "================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Running on Node     : $SLURMD_NODENAME"
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
if [ -d "$HOME/project/HPO/.venv" ]; then
    source "$HOME/project/HPO/.venv/bin/activate"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$HOME/.venv" ]; then
    source "$HOME/.venv/bin/activate"
fi

mkdir -p logs

# 2. Hardware Diagnostics
# ------------------------------------------------------------------------------
echo "Python Executable   : $(which python)"
python -c "import torch; print(f'PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
nvidia-smi

# 3. Concurrent Multi-Model Execution
# ------------------------------------------------------------------------------
# Prevent thread thrashing on CPU recurrent scan
export OMP_NUM_THREADS=4

echo ""
echo ">>> [1/3] Launching Model 11 (LightGBM) on CPU..."
python -u 18_hpo_lightgbm.py > "logs/11_lightgbm_${SLURM_JOB_ID}.log" 2>&1 &
PID_11=$!

echo ">>> [2/3] Launching Model 17 (S-Mamba) on H100 GPU..."
python -u 13_hpo_smamba_pytorch.py > "logs/17_smamba_${SLURM_JOB_ID}.log" 2>&1 &
PID_17=$!

echo ">>> [3/3] Launching Model 18 (PowerMamba) on H100 GPU..."
python -u 14_hpo_powermamba_pytorch.py > "logs/18_powermamba_${SLURM_JOB_ID}.log" 2>&1 &
PID_18=$!

echo ""
echo "All 3 processes successfully spawned:"
echo "  - PID $PID_11 : 18_hpo_lightgbm.py (CPU Multi-threading)"
echo "  - PID $PID_17 : 13_hpo_smamba_pytorch.py (H100 GPU)"
echo "  - PID $PID_18 : 14_hpo_powermamba_pytorch.py (H100 GPU)"
echo ""
echo "Waiting for all 3 models to complete..."

# 4. Wait for completion and capture exit codes
# ------------------------------------------------------------------------------
wait $PID_11
EXIT_11=$?
echo "Model 11 finished with exit code: $EXIT_11 at $(date)"

wait $PID_17
EXIT_17=$?
echo "Model 17 finished with exit code: $EXIT_17 at $(date)"

wait $PID_18
EXIT_18=$?
echo "Model 18 finished with exit code: $EXIT_18 at $(date)"

echo ""
echo "================================================================="
echo "All tasks finished at $(date)"
echo "================================================================="

# Exit with error if any model failed
if [ $EXIT_11 -ne 0 ] || [ $EXIT_17 -ne 0 ] || [ $EXIT_18 -ne 0 ]; then
    echo "[WARNING] One or more models exited with non-zero exit status."
    exit 1
fi

echo "[SUCCESS] All 3 models completed successfully!"
exit 0
