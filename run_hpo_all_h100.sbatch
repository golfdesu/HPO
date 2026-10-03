#!/bin/bash
# ==============================================================================
# SLURM Job Script: Master HPO Runner for ALL 22 Active Models on NVIDIA H100
# Target Hardware: compute4 node (NVIDIA H100 80GB HBM3, Erawan HPC)
# Partition      : gpu-h100 (Max timelimit: up to 5 days, requested: 24:00:00)
# Dataset Target : ACN Caltech V3 (../data_cleaned/acn_caltech_ready_v3.csv)
# Destination    : best_params/acn_caltech/
# ==============================================================================
#SBATCH --job-name=hpo_all_models_h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Concurrency: 4 models in parallel on NVIDIA H100 (80GB VRAM)
PARALLEL_JOBS=4

mkdir -p logs best_params/acn_caltech

PROGRESS_LOG="logs/hpo_progress_summary.txt"
COMPLETED_LOG="logs/hpo_completed.txt"
FAILED_LOG="logs/hpo_failed.txt"

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "Target Hardware     : NVIDIA H100 80GB HBM3 (Hopper, compute4)"
echo "Concurrency Mode    : PARALLEL_JOBS=${PARALLEL_JOBS}"
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
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
nvidia-smi

# 2. Execution Function with Tagged Output and Auto-Archival
run_model() {
    local script_name=$1
    local tag=$2
    local model_name=$3
    local json_out="${script_name%.py}_best_params.json"
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
        cp -v "${json_out}" best_params/acn_caltech/ 2>/dev/null || true
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

TOTAL_START=$(date +%s)

# 3. Catalog of Remaining 18 Models (Models 04 to 28; 00-03 already finished)
MODELS=(
    "04_hpo_ifm_pytorch.py:M04:04 Informer"
    "06_hpo_ptst_pytorch.py:M06:06 PatchTST"
    "07_hpo_itfm_pytorch.py:M07:07 iTransformer"
    "09_hpo_lstm_pytorch.py:M09:09 LSTM Baseline"
    "10_hpo_gru_pytorch.py:M10:10 GRU Baseline"
    "13_hpo_smamba_pytorch.py:M13:13 S-Mamba"
    "14_hpo_powermamba_pytorch.py:M14:14 PowerMamba"
    "15_hpo_timemachine_pytorch.py:M15:15 TimeMachine"
    "16_hpo_s4d_pytorch.py:M16:16 S4D Baseline"
    "17_hpo_xgboost.py:M17:17 XGBoost Direct"
    "18_hpo_lightgbm.py:M18:18 LightGBM Direct"
    "20_hpo_mft_pytorch.py:M20:20 Multi-Factor Transformer"
    "21_hpo_cnn_lstm_tfm_pytorch.py:M21:21 CNN-LSTM-Transformer"
    "22_hpo_fedformer_pytorch.py:M22:22 FEDformer"
    "23_hpo_tcn_pytorch.py:M23:23 TCN Baseline"
    "24_hpo_nhits_pytorch.py:M24:24 N-HiTS Baseline"
    "27_hpo_moderntcn_pytorch.py:M27:27 ModernTCN"
    "28_hpo_crossformer_pytorch.py:M28:28 Crossformer"
)

TOTAL_MODELS=${#MODELS[@]}
CURRENT_IDX=1

echo "Starting HPO execution of ${TOTAL_MODELS} active models on NVIDIA H100..."

if [ ${PARALLEL_JOBS} -gt 1 ]; then
    # Concurrent pool mode
    for entry in "${MODELS[@]}"; do
        IFS=":" read -r script_name model_tag model_name <<< "${entry}"
        run_model "${script_name}" "${model_tag}" "${model_name}" &
        while [ $(jobs -r -p | wc -l) -ge ${PARALLEL_JOBS} ]; do
            wait -n 2>/dev/null || sleep 3
        done
    done
    wait
else
    # Sequential execution mode
    for entry in "${MODELS[@]}"; do
        IFS=":" read -r script_name model_tag model_name <<< "${entry}"
        echo "[Queue Progress: ${CURRENT_IDX}/${TOTAL_MODELS}] Starting ${model_name}..."
        run_model "${script_name}" "${model_tag}" "${model_name}"
        CURRENT_IDX=$((CURRENT_IDX + 1))
    done
fi

TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$(awk "BEGIN {printf \"%.2f\", ${TOTAL_DIFF}/3600}")

echo ""
echo "=============================================================================="
echo "🎯 MASTER HPO RUN COMPLETE ON NVIDIA H100"
echo "Total Walltime      : ${TOTAL_HOURS} hours (${TOTAL_DIFF} seconds)"
echo "Finished At         : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Successful Models   : $(wc -l < "${COMPLETED_LOG}" 2>/dev/null || echo 0) / ${TOTAL_MODELS}"
echo "Failed Models       : $(wc -l < "${FAILED_LOG}" 2>/dev/null || echo 0) / ${TOTAL_MODELS}"
echo "Archived Best Params: $(ls -1 best_params/acn_caltech/*_best_params.json 2>/dev/null | wc -l) files"
echo "=============================================================================="
