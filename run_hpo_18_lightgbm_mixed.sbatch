#!/bin/bash
# ==============================================================================
# SLURM Job Script: Dedicated Model 18 (LightGBM) HPO Runner on 'mixed' Partition
# Target Model   : 18_hpo_lightgbm.py (Direct Multi-Step GBDT)
# Hardware Target: Erawan HPC 'mixed' partition (16 CPU cores, 32GB RAM)
# Optimization   : Optimized CPU Multi-Threading (OMP_NUM_THREADS=16, n_jobs=-1)
# Output JSON    : 18_hpo_lightgbm_best_params.json (Saved to root and archived)
# ==============================================================================
#SBATCH --job-name=hpo_18_lgbm_mixed
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p logs best_params/acn_caltech

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "Target Model        : 18_hpo_lightgbm.py"
echo "Assigned CPUs       : 16 Cores"
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
elif [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
else
    echo "Warning: No virtualenv found, using default python: $(which python)"
fi

echo "Python Executable   : $(which python)"

# 2. CPU Multi-Threading Optimizations for LightGBM
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1

# 3. Execute LightGBM HPO
SCRIPT_NAME="18_hpo_lightgbm.py"
JSON_OUT="18_hpo_lightgbm_best_params.json"
START_T=$(date +%s)

echo ""
echo "=============================================================================="
echo ">>> STARTING LIGHTGBM HPO SEARCH (50 TRIALS, EVAL_STEPS=[0, 5, 11, 23, 35, 47])"
echo "=============================================================================="

python -u "${SCRIPT_NAME}"
EXIT_CODE=$?

END_T=$(date +%s)
DIFF_T=$((END_T - START_T))
ELAPSED_MIN=$((DIFF_T / 60))

echo ""
echo "=============================================================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> [SUCCESS] LightGBM HPO COMPLETED in ${ELAPSED_MIN}m (${DIFF_T}s)"
    if [ -f "${JSON_OUT}" ]; then
        echo "Primary JSON Output: $(pwd)/${JSON_OUT}"
        cp -v "${JSON_OUT}" best_params/acn_caltech/ 2>/dev/null || true
    fi
else
    echo ">>> [FAILED] LightGBM HPO FAILED with exit code ${EXIT_CODE} after ${ELAPSED_MIN}m"
fi
echo "=============================================================================="
echo "Job Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
