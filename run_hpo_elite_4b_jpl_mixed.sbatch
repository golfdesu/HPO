#!/bin/bash
# ==============================================================================
# SLURM Job Script: HPO Runner for Elite Baselines (Group B) on JPL V3 (mixed)
# Target Hardware: Erawan HPC 'mixed' partition (compute3, A100, 16 CPU, 64GB RAM)
# Partition      : mixed (Max timelimit: 1 day, requested: 24:00:00)
# Dataset Target : ACN JPL V3 (acn_jpl_ready_v3.csv)
# Destination    : Root directory (.) strictly, suffixed with _jpl_v3
#                  e.g. 01_hpo_tfm_pytorch_best_params_jpl_v3.json
#                  (never collides with Caltech <model>_best_params.json)
# Models Included: 4 Elite Baselines (17, 24, 27, 28)
# Companion Job  : run_hpo_elite_4a_jpl_h100.sbatch (models 01, 06, 10, 13)
# ==============================================================================
#SBATCH --job-name=hpo_elite_4b_jpl_mixed
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Concurrency Mode: 2 parallel jobs (safe on mixed GPU, same as run_hpo_elite_mixed.sbatch)
# Set to 1 for strictly sequential execution.
PARALLEL_JOBS=2

mkdir -p logs

export DATASET_TARGET="acn_jpl_ready_v3"
# Output suffix consumed by the HPO scripts -> <model>_best_params_jpl_v3.json
export HPO_OUTPUT_SUFFIX="_jpl_v3"

PROGRESS_LOG="logs/hpo_progress_jpl_elite_4b_mixed.txt"
COMPLETED_LOG="logs/hpo_completed_jpl_elite_4b_mixed.txt"
FAILED_LOG="logs/hpo_failed_jpl_elite_4b_mixed.txt"

rm -f "${PROGRESS_LOG}" "${COMPLETED_LOG}" "${FAILED_LOG}"

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "Concurrency Mode    : PARALLEL_JOBS=${PARALLEL_JOBS}"
echo "Target Dataset      : ACN JPL V3 (${DATASET_TARGET})"
echo "Output Suffix       : ${HPO_OUTPUT_SUFFIX}"
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

# 2. Multithreading & CUDA Environment Optimizations (mixed / A100)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Running in CPU mode')"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

TOTAL_START=$(date +%s)

# 3. Helper Function to Run HPO (output stays in root, no subdirectories)
run_hpo() {
    local script_name=$1
    local tag=$2
    local model_name=$3
    local json_out="${script_name%.py}_best_params${HPO_OUTPUT_SUFFIX}.json"
    local start_t=$(date +%s)

    echo ""
    echo "=============================================================================="
    echo ">>> [STARTED]  [${tag}] ${model_name} (${script_name}) at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================================="

    if [ ${PARALLEL_JOBS} -gt 1 ]; then
        python -u "${script_name}" 2>&1 | sed -u "s/^/[${tag}] /"
        local exit_code=${PIPESTATUS[0]}
    else
        python -u "${script_name}"
        local exit_code=$?
    fi

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

# 4. Catalog of 4 Elite Baselines (Group B: 17, 24, 27, 28)
MODELS=(
    "17_hpo_xgboost.py:M17:17 XGBoost Direct Baseline"
    "24_hpo_nhits_pytorch.py:M24:24 N-HiTS Baseline"
    "27_hpo_moderntcn_pytorch.py:M27:27 ModernTCN Baseline"
    "28_hpo_crossformer_pytorch.py:M28:28 Crossformer Baseline"
)

TOTAL_MODELS=${#MODELS[@]}
CURRENT_IDX=1

echo "=============================================================================="
echo "Starting HPO for ${TOTAL_MODELS} Elite Models (Group B) on JPL V3 using 'mixed' partition..."
echo "Concurrency: PARALLEL_JOBS=${PARALLEL_JOBS}"
echo "=============================================================================="

if [ ${PARALLEL_JOBS} -gt 1 ]; then
    for entry in "${MODELS[@]}"; do
        IFS=":" read -r script_name model_tag model_name <<< "${entry}"
        run_hpo "${script_name}" "${model_tag}" "${model_name}" &
        while [ $(jobs -r -p | wc -l) -ge ${PARALLEL_JOBS} ]; do
            wait -n 2>/dev/null || sleep 3
        done
    done
    wait
else
    for entry in "${MODELS[@]}"; do
        IFS=":" read -r script_name model_tag model_name <<< "${entry}"
        echo "[Queue Progress: ${CURRENT_IDX}/${TOTAL_MODELS}] Starting ${model_name}..."
        run_hpo "${script_name}" "${model_tag}" "${model_name}"
        CURRENT_IDX=$((CURRENT_IDX + 1))
    done
fi

TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_DIFF / 3600))
TOTAL_MIN=$(((TOTAL_DIFF % 3600) / 60))

echo "=============================================================================="
echo ">>> HPO COMPLETED FOR ${TOTAL_MODELS} ELITE MODELS (GROUP B) ON JPL V3 (mixed)"
echo "Total Elapsed Time  : ${TOTAL_HOURS}h ${TOTAL_MIN}m ($TOTAL_DIFF seconds)"
echo "Finished at         : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
