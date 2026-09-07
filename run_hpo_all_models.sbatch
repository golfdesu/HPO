#!/bin/bash
# ==============================================================================
# SLURM Job Script: Master HPO Runner for ALL 32 Models on 'mixed' Partition
# Target Models: 00 to 31 (All Architecture Families)
# Hardware Target: Erawan HPC 'mixed' partition (1x GPU, 8 CPUs, 32GB RAM)
# Estimated Runtime: ~8.0 - 14.0 hours total (50 trials each, 30 epochs, patience 10)
# ==============================================================================
#SBATCH --job-name=hpo_all_32_models
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

mkdir -p logs best_params

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
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

# 2. Execution Function
run_hpo() {
    local step_num=$1
    local script_name=$2
    local model_name=$3

    echo ""
    echo "=============================================================================="
    echo ">>> [${step_num}] STARTING HPO: ${model_name} (${script_name})"
    echo ">>> Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================================="

    local start_t=$(date +%s)
    python -u "${script_name}"
    local exit_code=$?
    local end_t=$(date +%s)
    local diff_t=$((end_t - start_t))

    local json_out="${script_name%.py}_best_params.json"
    if [ -f "${json_out}" ]; then
        cp -v "${json_out}" best_params/ 2>/dev/null || true
    fi

    if [ ${exit_code} -eq 0 ]; then
        echo ">>> [${step_num}] [SUCCESS]: ${model_name} completed in ${diff_t}s ($((diff_t / 60)) min)"
    else
        echo ">>> [${step_num}] [ERROR]: ${model_name} FAILED with exit code ${exit_code} after ${diff_t}s"
    fi
    return ${exit_code}
}

TOTAL_START=$(date +%s)
FAILED_MODELS=()

# All 32 Models Catalog (00 to 31)
MODELS=(
    "00_hpo_tfm_custom_pytorch.py:00 Custom Transformer"
    "01_hpo_tfm_pytorch.py:01 Vanilla Transformer Encoder"
    "02_hpo_dec_pytorch.py:02 Vanilla Transformer Decoder"
    "03_hpo_encdec_pytorch.py:03 Seq2Seq Transformer"
    "04_hpo_ifm_pytorch.py:04 Informer"
    "05_hpo_afm_pytorch.py:05 Autoformer"
    "06_hpo_ptst_pytorch.py:06 PatchTST"
    "07_hpo_itfm_pytorch.py:07 iTransformer"
    "08_hpo_timesnet_pytorch.py:08 TimesNet"
    "09_hpo_lstm_pytorch.py:09 LSTM Baseline"
    "10_hpo_gru_pytorch.py:10 GRU Baseline"
    "11_hpo_dlinear_pytorch.py:11 DLinear"
    "12_hpo_nlinear_pytorch.py:12 NLinear"
    "13_hpo_smamba_pytorch.py:13 S-Mamba"
    "14_hpo_powermamba_pytorch.py:14 PowerMamba"
    "15_hpo_timemachine_pytorch.py:15 TimeMachine"
    "16_hpo_s4d_pytorch.py:16 S4D Baseline"
    "17_hpo_xgboost.py:17 XGBoost Direct"
    "18_hpo_lightgbm.py:18 LightGBM Direct"
    "19_hpo_sarima.py:19 SARIMA Baseline"
    "20_hpo_mft_pytorch.py:20 Multi-Factor Transformer"
    "21_hpo_cnn_lstm_tfm_pytorch.py:21 CNN-LSTM-Transformer"
    "22_hpo_fedformer_pytorch.py:22 FEDformer"
    "23_hpo_tcn_pytorch.py:23 TCN Baseline"
    "24_hpo_nhits_pytorch.py:24 N-HiTS Baseline"
    "25_hpo_tide_pytorch.py:25 TiDE Baseline"
    "26_hpo_nbeats_pytorch.py:26 N-BEATS Baseline"
    "27_hpo_moderntcn_pytorch.py:27 ModernTCN"
    "28_hpo_crossformer_pytorch.py:28 Crossformer"
    "29_hpo_segrnn_pytorch.py:29 SegRNN"
    "30_hpo_nstransformer_pytorch.py:30 Non-stationary Transformer"
    "31_hpo_scinet_pytorch.py:31 SCINet"
)

TOTAL_MODELS=${#MODELS[@]}
CURRENT_STEP=1

for entry in "${MODELS[@]}"; do
    IFS=":" read -r script_name model_name <<< "${entry}"
    run_hpo "${CURRENT_STEP}/${TOTAL_MODELS}" "${script_name}" "${model_name}"
    if [ $? -ne 0 ]; then
        FAILED_MODELS+=("${model_name}")
    fi
    CURRENT_STEP=$((CURRENT_STEP + 1))
done

TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))

echo ""
echo "=============================================================================="
echo "ALL 32 HPO SEARCHES COMPLETE"
echo "Total Elapsed Time: ${TOTAL_DIFF}s ($((TOTAL_DIFF / 60)) min)"
if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
    echo "Status: ALL 32 MODELS COMPLETED SUCCESSFULLY"
else
    echo "Status: COMPLETED WITH FAILURES"
    echo "Failed Models: ${FAILED_MODELS[*]}"
fi
echo "=============================================================================="
