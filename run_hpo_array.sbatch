#!/bin/bash
# ==============================================================================
# SLURM Job Array: Parallel Runner for Files 17-20 on Erawan HPC (ThaiSC)
# Runs files 17, 18, 19, 20 concurrently (if GPUs are available in cluster queue)
# ==============================================================================
#SBATCH --job-name=hpo_ssm
#SBATCH --output=logs/hpo_%a_%A.out
#SBATCH --error=logs/hpo_%a_%A.err
#SBATCH --partition=gpu-h100,mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=17-20%4

# %4 limits max concurrent running tasks to 4 (can change to %2 or %1 if quota restricted)

echo "================================================================="
echo "SLURM Array Job ID  : $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on Node     : $SLURMD_NODENAME"
echo "Start Time          : $(date)"
echo "================================================================="

# 1. Environment & Module Setup
# ------------------------------------------------------------------------------
if command -v module &> /dev/null; then
    module purge
    module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi

if [ -d "$HOME/project/HPO/.venv" ]; then
    source "$HOME/project/HPO/.venv/bin/activate"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$HOME/.venv" ]; then
    source "$HOME/.venv/bin/activate"
fi

mkdir -p logs

# 2. Select Script based on Array Task ID
# ------------------------------------------------------------------------------
case $SLURM_ARRAY_TASK_ID in
    17) SCRIPT="17_hpo_smamba_pytorch.py" ;;
    18) SCRIPT="18_hpo_powermamba_pytorch.py" ;;
    19) SCRIPT="19_hpo_timemachine_pytorch.py" ;;
    20) SCRIPT="20_hpo_s4d_pytorch.py" ;;
    *)
        echo "[ERROR] Unknown SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "Executing: python -u $SCRIPT"
nvidia-smi

python -u "$SCRIPT"
EXIT_CODE=$?

echo ""
echo "Finished $SCRIPT with exit code $EXIT_CODE at $(date)"
exit $EXIT_CODE
