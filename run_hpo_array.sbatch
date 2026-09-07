#!/bin/bash
# ==============================================================================
# SLURM Job Array: Parallel HPO Runner for ALL 31 Baselines on 'mixed' Partition
# Target Models: 01 to 31 (All Baseline Architecture Families)
# Hardware Target: Erawan HPC 'mixed' partition (1x GPU, 8 CPUs, 32GB RAM per task)
# Walltime Limit: 48:00:00 (2 Days)
# Concurrency: All 31 models dispatched in parallel
# ==============================================================================
#SBATCH --job-name=hpo_array_31
#SBATCH --output=logs/hpo_%a_%A.out
#SBATCH --error=logs/hpo_%a_%A.err
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=1-31

mkdir -p logs best_params/acn_jpl

echo "=============================================================================="
echo "SLURM Job Array ID  : $SLURM_ARRAY_JOB_ID"
echo "SLURM Task ID       : $SLURM_ARRAY_TASK_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
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

# 2. Model Lookup Table (Task ID 1 to 31)
MODELS=(
    [1]="01_hpo_tfm_pytorch.py:01 Vanilla Transformer Encoder"
    [2]="02_hpo_dec_pytorch.py:02 Vanilla Transformer Decoder"
    [3]="03_hpo_encdec_pytorch.py:03 Seq2Seq Transformer"
    [4]="04_hpo_ifm_pytorch.py:04 Informer"
    [5]="05_hpo_afm_pytorch.py:05 Autoformer"
    [6]="06_hpo_ptst_pytorch.py:06 PatchTST"
    [7]="07_hpo_itfm_pytorch.py:07 iTransformer"
    [8]="08_hpo_timesnet_pytorch.py:08 TimesNet"
    [9]="09_hpo_lstm_pytorch.py:09 LSTM Baseline"
    [10]="10_hpo_gru_pytorch.py:10 GRU Baseline"
    [11]="11_hpo_dlinear_pytorch.py:11 DLinear"
    [12]="12_hpo_nlinear_pytorch.py:12 NLinear"
    [13]="13_hpo_smamba_pytorch.py:13 S-Mamba"
    [14]="14_hpo_powermamba_pytorch.py:14 PowerMamba"
    [15]="15_hpo_timemachine_pytorch.py:15 TimeMachine"
    [16]="16_hpo_s4d_pytorch.py:16 S4D Baseline"
    [17]="17_hpo_xgboost.py:17 XGBoost Direct"
    [18]="18_hpo_lightgbm.py:18 LightGBM Direct"
    [19]="19_hpo_sarima.py:19 SARIMA Baseline"
    [20]="20_hpo_mft_pytorch.py:20 Multi-Factor Transformer"
    [21]="21_hpo_cnn_lstm_tfm_pytorch.py:21 CNN-LSTM-Transformer"
    [22]="22_hpo_fedformer_pytorch.py:22 FEDformer"
    [23]="23_hpo_tcn_pytorch.py:23 TCN Baseline"
    [24]="24_hpo_nhits_pytorch.py:24 N-HiTS Baseline"
    [25]="25_hpo_tide_pytorch.py:25 TiDE Baseline"
    [26]="26_hpo_nbeats_pytorch.py:26 N-BEATS Baseline"
    [27]="27_hpo_moderntcn_pytorch.py:27 ModernTCN"
    [28]="28_hpo_crossformer_pytorch.py:28 Crossformer"
    [29]="29_hpo_segrnn_pytorch.py:29 SegRNN"
    [30]="30_hpo_nstransformer_pytorch.py:30 Non-stationary Transformer"
    [31]="31_hpo_scinet_pytorch.py:31 SCINet"
)

TARGET_ENTRY="${MODELS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$TARGET_ENTRY" ]; then
    echo "[ERROR] Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID (Valid range: 1-31)"
    exit 1
fi

IFS=":" read -r SCRIPT_NAME MODEL_NAME <<< "$TARGET_ENTRY"

echo ""
echo "=============================================================================="
echo ">>> EXECUTING TASK $SLURM_ARRAY_TASK_ID / 31: $MODEL_NAME"
echo ">>> Script: $SCRIPT_NAME"
echo ">>> Dataset: acn_jpl_ready.csv"
echo "=============================================================================="

START_T=$(date +%s)
python -u "$SCRIPT_NAME"
EXIT_CODE=$?
END_T=$(date +%s)
DIFF_T=$((END_T - START_T))

# Archive result into best_params/acn_jpl/
JSON_OUT="${SCRIPT_NAME%.py}_best_params.json"
if [ -f "${JSON_OUT}" ]; then
    cp -v "${JSON_OUT}" best_params/acn_jpl/ 2>/dev/null || true
fi

echo ""
echo "=============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo ">>> [SUCCESS] $MODEL_NAME finished in ${DIFF_T}s ($((DIFF_T / 60)) min) with exit code 0"
else
    echo ">>> [ERROR] $MODEL_NAME FAILED with exit code $EXIT_CODE after ${DIFF_T}s"
fi
echo "=============================================================================="
exit $EXIT_CODE
