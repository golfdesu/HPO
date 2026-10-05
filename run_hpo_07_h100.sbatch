#!/bin/bash
# ==============================================================================
# SLURM Job Script: Dedicated HPO Runner for Model 07 (iTransformer) on NVIDIA H100
# Target Hardware: compute4 node (NVIDIA H100 80GB HBM3, Erawan HPC)
# Partition      : gpu-h100 (Max timelimit: up to 5 days, requested: 12:00:00)
# Dataset Target : ACN JPL/JPN V3 (../data_cleaned/acn_jpl_ready_v3.csv)
# Destination    : best_params/acn_jpl/
# Model Included : 07_hpo_itfm_pytorch.py (iTransformer ICLR 2024 Spotlight)
# ==============================================================================
#SBATCH --job-name=hpo_07_itfm_h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

mkdir -p logs best_params/acn_jpl

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "Target Hardware     : NVIDIA H100 80GB HBM3 (Hopper, compute4)"
echo "Target Dataset      : ACN JPL/JPN V3 (acn_jpl_ready_v3.csv)"
echo "Target Model        : Model 07 iTransformer (07_hpo_itfm_pytorch.py)"
echo "Destination Dir     : best_params/acn_jpl/"
echo "Unified Log File    : logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "=============================================================================="

# 1. Environment & CUDA Setup
if command -v module &> /dev/null; then
    module purge
    module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi

# Activate Virtual Environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "../environment/.venv/bin/activate" ]; then
    source ../environment/.venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
elif [ -d "$HOME/project/HPO/.venv" ]; then
    source "$HOME/project/HPO/.venv/bin/activate"
elif [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
else
    echo "Warning: No virtualenv found, using default python: $(which python)"
fi

echo "Python Executable   : $(which python)"

# 2. Multithreading & CUDA Environment Optimizations
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Running in CPU mode')"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

# 3. Execute Model 07 HPO
SCRIPT="07_hpo_itfm_pytorch.py"
JSON_OUT="07_hpo_itfm_pytorch_best_params.json"
START_T=$(date +%s)

echo ""
echo "=============================================================================="
echo ">>> [STARTED] Model 07 iTransformer (${SCRIPT}) at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

python -u "${SCRIPT}"
EXIT_CODE=$?

END_T=$(date +%s)
DIFF_T=$((END_T - START_T))
ELAPSED_MIN=$((DIFF_T / 60))

if [ -f "${JSON_OUT}" ]; then
    cp -v "${JSON_OUT}" best_params/acn_jpl/ 2>/dev/null || true
    echo "Successfully confirmed ${JSON_OUT} is at root: $(pwd)/${JSON_OUT}"
fi

echo ""
echo "=============================================================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> [FINISHED] Model 07 iTransformer COMPLETED in ${ELAPSED_MIN}m (${DIFF_T}s)"
    echo "Saved output to ${JSON_OUT} and best_params/acn_jpl/${JSON_OUT}"
else
    echo ">>> [FAILED] Model 07 iTransformer FAILED with exit code ${EXIT_CODE} after ${ELAPSED_MIN}m"
fi
echo "=============================================================================="
