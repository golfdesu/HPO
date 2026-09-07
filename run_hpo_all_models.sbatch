#!/bin/bash
# ==============================================================================
# SLURM Job Script: Master HPO Runner for ALL 31 Baselines on 'mixed' Partition
# Concurrency: 4 Parallel Models on 1x NVIDIA A100 GPU with Tagged Unified Output
# Target Models: 01 to 31 (All Architecture Families)
# Hardware Target: Erawan HPC 'mixed' partition (1x A100 GPU, 8 CPUs, 32GB RAM)
# Estimated Runtime: ~3.5 - 5.0 hours total (4 models concurrent)
# ==============================================================================
#SBATCH --job-name=hpo_baselines_31_models
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

PARALLEL_JOBS=4

mkdir -p logs best_params/acn_jpl
rm -f logs/hpo_completed.txt logs/hpo_failed.txt logs/hpo_progress_summary.txt

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "Concurrency Limit   : ${PARALLEL_JOBS} models parallel on 1 GPU"
echo "Unified Log File    : logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "=============================================================================="

# 1. Environment & Module Setup
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
else
    echo "Warning: No virtualenv found, using default python: $(which python)"
fi

echo "Python Executable   : $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
nvidia-smi

# 2. Execution Function with Tagged Unified Output
run_model() {
    local script_name=$1
    local tag=$2
    local model_name=$3
    local json_out="${script_name%.py}_best_params.json"
    local start_t=$(date +%s)

    echo ""
    echo "=============================================================================="
    echo ">>> [STARTED]  [${tag}] ${model_name} at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================================="

    # Run unbuffered and prefix each stdout/stderr line with [tag] into unified SLURM log
    python -u "${script_name}" 2>&1 | sed -u "s/^/[${tag}] /"
    local exit_code=${PIPESTATUS[0]}
    local end_t=$(date +%s)
    local diff_t=$((end_t - start_t))
    local elapsed_min=$((diff_t / 60))

    if [ -f "${json_out}" ]; then
        cp -v "${json_out}" best_params/acn_jpl/ 2>/dev/null || true
    fi

    echo ""
    echo "=============================================================================="
    if [ ${exit_code} -eq 0 ]; then
        echo ">>> [FINISHED] [${tag}] ${model_name} COMPLETED in ${elapsed_min}m (${diff_t}s)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] [${tag}] ${model_name} (${elapsed_min}m)" >> logs/hpo_progress_summary.txt
        echo "${tag}: ${model_name}" >> logs/hpo_completed.txt
    else
        echo ">>> [FAILED]   [${tag}] ${model_name} FAILED with exit code ${exit_code} after ${elapsed_min}m"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FAILED ] [${tag}] ${model_name} (Exit ${exit_code}, ${elapsed_min}m)" >> logs/hpo_progress_summary.txt
        echo "${tag}: ${model_name} (Exit ${exit_code})" >> logs/hpo_failed.txt
    fi
    echo "=============================================================================="
    echo ""
}

TOTAL_START=$(date +%s)

MODELS=(
    "01_hpo_tfm_pytorch.py:M01:01 Vanilla Transformer Encoder"
    "02_hpo_dec_pytorch.py:M02:02 Vanilla Transformer Decoder"
    "03_hpo_encdec_pytorch.py:M03:03 Seq2Seq Transformer"
    "04_hpo_ifm_pytorch.py:M04:04 Informer"
    "05_hpo_afm_pytorch.py:M05:05 Autoformer"
    "06_hpo_ptst_pytorch.py:M06:06 PatchTST"
    "07_hpo_itfm_pytorch.py:M07:07 iTransformer"
    "08_hpo_timesnet_pytorch.py:M08:08 TimesNet"
    "09_hpo_lstm_pytorch.py:M09:09 LSTM Baseline"
    "10_hpo_gru_pytorch.py:M10:10 GRU Baseline"
    "11_hpo_dlinear_pytorch.py:M11:11 DLinear"
    "12_hpo_nlinear_pytorch.py:M12:12 NLinear"
    "13_hpo_smamba_pytorch.py:M13:13 S-Mamba"
    "14_hpo_powermamba_pytorch.py:M14:14 PowerMamba"
    "15_hpo_timemachine_pytorch.py:M15:15 TimeMachine"
    "16_hpo_s4d_pytorch.py:M16:16 S4D Baseline"
    "17_hpo_xgboost.py:M17:17 XGBoost Direct"
    "18_hpo_lightgbm.py:M18:18 LightGBM Direct"
    "19_hpo_sarima.py:M19:19 SARIMA Baseline"
    "20_hpo_mft_pytorch.py:M20:20 Multi-Factor Transformer"
    "21_hpo_cnn_lstm_tfm_pytorch.py:M21:21 CNN-LSTM-Transformer"
    "22_hpo_fedformer_pytorch.py:M22:22 FEDformer"
    "23_hpo_tcn_pytorch.py:M23:23 TCN Baseline"
    "24_hpo_nhits_pytorch.py:M24:24 N-HiTS Baseline"
    "25_hpo_tide_pytorch.py:M25:25 TiDE Baseline"
    "26_hpo_nbeats_pytorch.py:M26:26 N-BEATS Baseline"
    "27_hpo_moderntcn_pytorch.py:M27:27 ModernTCN"
    "28_hpo_crossformer_pytorch.py:M28:28 Crossformer"
    "29_hpo_segrnn_pytorch.py:M29:29 SegRNN"
    "30_hpo_nstransformer_pytorch.py:M30:30 Non-stationary Transformer"
    "31_hpo_scinet_pytorch.py:M31:31 SCINet"
)

# Launch with worker pool maintaining PARALLEL_JOBS concurrent processes
for entry in "${MODELS[@]}"; do
    IFS=":" read -r script_name model_tag model_name <<< "${entry}"
    
    run_model "${script_name}" "${model_tag}" "${model_name}" &
    
    while [ $(jobs -r -p | wc -l) -ge ${PARALLEL_JOBS} ]; do
        wait -n 2>/dev/null || sleep 3
    done
done

# Wait for all remaining background jobs to finish
wait

TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))
TOTAL_MIN=$((TOTAL_DIFF / 60))

echo ""
echo "=============================================================================="
echo "🎉 ALL 31 BASELINE HPO SEARCHES COMPLETE"
echo "Total Elapsed Time: ${TOTAL_DIFF}s (${TOTAL_MIN} min)"
echo "------------------------------------------------------------------------------"
echo "Completed Models Summary:"
if [ -f logs/hpo_completed.txt ]; then
    cat logs/hpo_completed.txt
fi
echo "------------------------------------------------------------------------------"
if [ -f logs/hpo_failed.txt ] && [ -s logs/hpo_failed.txt ]; then
    echo "⚠️ Status: COMPLETED WITH FAILURES"
    echo "Failed Models:"
    cat logs/hpo_failed.txt
else
    echo "✅ Status: ALL 31 BASELINES COMPLETED SUCCESSFULLY!"
fi
echo "=============================================================================="
