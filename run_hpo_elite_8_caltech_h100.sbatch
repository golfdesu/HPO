#!/bin/bash
# ==============================================================================
# SLURM Job Script: Dedicated HPO Runner for 8 Elite Baselines on Caltech V3 (H100)
# Target Hardware: compute4 node (NVIDIA H100 80GB HBM3, Erawan HPC)
# Partition      : gpu-h100 (Max timelimit: up to 5 days, requested: 24:00:00)
# Dataset Target : ACN Caltech V3 (acn_caltech_ready_v3.csv)
# Destination    : Root directory (.) strictly
# Models Included: 8 Elite Baselines (01, 06, 10, 13, 17, 24, 27, 28)
# ==============================================================================
#SBATCH --job-name=hpo_elite_8_caltech_h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

mkdir -p logs

export DATASET_TARGET="acn_caltech_ready_v3"

PROGRESS_LOG="logs/hpo_progress_caltech_elite_8_h100.txt"
COMPLETED_LOG="logs/hpo_completed_caltech_elite_8_h100.txt"
FAILED_LOG="logs/hpo_failed_caltech_elite_8_h100.txt"

rm -f "${PROGRESS_LOG}" "${COMPLETED_LOG}" "${FAILED_LOG}"

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "Target Hardware     : NVIDIA H100 80GB HBM3 (Hopper, compute4)"
echo "Target Dataset      : ACN Caltech V3 (${DATASET_TARGET})"
echo "Destination Dir     : Root directory ($(pwd))"
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

# 2. Strict Hardware Assertion: NVIDIA H100 Verification
echo "------------------------------------------------------------------------------"
echo "Checking GPU Hardware..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
    echo "Detected Primary GPU: ${GPU_NAME}"
    if [[ "${GPU_NAME}" != *"H100"* ]]; then
        echo "❌ FATAL ERROR: Detected GPU '${GPU_NAME}' is NOT an NVIDIA H100!"
        echo "The job configuration strictly enforces execution on an H100 node (compute4)."
        echo "Aborting job to maintain benchmark fidelity."
        exit 1
    fi
    echo "✅ Confirmed: Running strictly on NVIDIA H100 80GB HBM3."
fi
echo "------------------------------------------------------------------------------"

# 3. High-Performance Hopper & H100 Optimizations
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Running in CPU mode')"

TOTAL_START=$(date +%s)

# 4. Helper Function to Run and Archive HPO
run_hpo() {
    local script_name=$1
    local tag=$2
    local model_name=$3
    local json_out="${script_name%.py}_best_params.json"
    local start_t=$(date +%s)

    echo ""
    echo "=============================================================================="
    echo ">>> [STARTED]  [${tag}] ${model_name} (${script_name}) at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================================="

    python -u "${script_name}"
    local exit_code=$?
    local end_t=$(date +%s)
    local diff_t=$((end_t - start_t))
    local elapsed_min=$((diff_t / 60))

    if [ -f "${json_out}" ]; then
        echo ">>> [SAVED] Confirmed in root: $(pwd)/${json_out}"
    fi

    echo ""
    echo "=============================================================================="
    if [ ${exit_code} -eq 0 ]; then
        echo ">>> [FINISHED] [${tag}] ${model_name} COMPLETED in ${elapsed_min}m (${diff_t}s)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] [${tag}] ${model_name} (${elapsed_min}m)" >> "${PROGRESS_LOG}"
        echo "${tag}: ${model_name} - SUCCESS (${elapsed_min}m)" >> "${COMPLETED_LOG}"
    else
        echo ">>> [FAILED]   [${tag}] ${model_name} FAILED with exit code ${exit_code} after ${elapsed_min}m"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FAILED ] [${tag}] ${model_name} (Exit ${exit_code}, ${elapsed_min}m)" >> "${PROGRESS_LOG}"
        echo "${tag}: ${model_name} - FAILED (Exit ${exit_code})" >> "${FAILED_LOG}"
    fi
    echo "=============================================================================="
    echo ""
}

# 5. Catalog of 8 Elite Baselines (01, 06, 10, 13, 17, 24, 27, 28)
MODELS=(
    "01_hpo_tfm_pytorch.py:M01:01 Vanilla Transformer Encoder"
    "06_hpo_ptst_pytorch.py:M06:06 PatchTST Baseline"
    "10_hpo_gru_pytorch.py:M10:10 GRU Baseline"
    "13_hpo_smamba_pytorch.py:M13:13 S-Mamba Baseline"
    "17_hpo_xgboost.py:M17:17 XGBoost Direct Baseline"
    "24_hpo_nhits_pytorch.py:M24:24 N-HiTS Baseline"
    "27_hpo_moderntcn_pytorch.py:M27:27 ModernTCN Baseline"
    "28_hpo_crossformer_pytorch.py:M28:28 Crossformer Baseline"
)

TOTAL_MODELS=${#MODELS[@]}
CURRENT_IDX=1

echo "=============================================================================="
echo "Starting HPO for ${TOTAL_MODELS} Elite Models on Caltech V3 using NVIDIA H100..."
echo "=============================================================================="

for entry in "${MODELS[@]}"; do
    IFS=":" read -r script_name model_tag model_name <<< "${entry}"
    echo "[Queue Progress: ${CURRENT_IDX}/${TOTAL_MODELS}] Starting ${model_name}..."
    run_hpo "${script_name}" "${model_tag}" "${model_name}"
    CURRENT_IDX=$((CURRENT_IDX + 1))
done

TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_DIFF / 3600))
TOTAL_MIN=$(((TOTAL_DIFF % 3600) / 60))

echo "=============================================================================="
echo ">>> HPO COMPLETED FOR ALL ${TOTAL_MODELS} ELITE MODELS ON CALTECH V3 (NVIDIA H100)"
echo "Total Elapsed Time  : ${TOTAL_HOURS}h ${TOTAL_MIN}m ($TOTAL_DIFF seconds)"
echo "Finished at         : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
