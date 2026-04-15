#!/bin/bash
#SBATCH --job-name=mlebench-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --qos=h200_agentic-models_high
#SBATCH --output=/checkpoint/agentic-models/%u/slurm_logs/%j.out
#SBATCH --error=/checkpoint/agentic-models/%u/slurm_logs/%j.err

# MLE-bench Evaluation SLURM Launcher (Simple)
#
# Usage:
#   # Single task
#   sbatch --export=TASK=mlsp-2013-birds launch.sh
#
#   # With custom samples
#   sbatch --export=TASK=mlsp-2013-birds,SAMPLES=64 launch.sh
#
#   # With custom config
#   sbatch --export=TASK=mlsp-2013-birds,CONFIG=configs/gpt5.yaml launch.sh
#
#   # Multiple tasks (comma-separated)
#   sbatch --export=TASKS=mlsp-2013-birds,spooky-author-identification launch.sh

# Ensure log directory exists
mkdir -p /checkpoint/agentic-models/${USER}/slurm_logs

# Activate conda environment
source /home/winnieyangwn/miniconda3/etc/profile.d/conda.sh
conda activate rllm

# Change to rllm directory
cd /home/winnieyangwn/rllm

# Set defaults
CONFIG=${CONFIG:-/home/winnieyangwn/rllm/cookbooks/mlebench/eval_integration/configs/gpt5.yaml}
SAMPLES=${SAMPLES:-64}
OUTPUT_DIR=${OUTPUT_DIR:-/checkpoint/maui_sft/winnieyangwn/RLLM/slurm_${SLURM_JOB_ID}}

echo "========================================"
echo "MLE-bench Evaluation Job: ${SLURM_JOB_ID}"
echo "========================================"
echo "Config: ${CONFIG}"
echo "Samples: ${SAMPLES}"
echo "Output: ${OUTPUT_DIR}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

# Run evaluation
if [ -n "${TASK}" ]; then
    # Single task mode
    echo "Running single task: ${TASK}"
    python /home/winnieyangwn/rllm/cookbooks/mlebench/eval_integration/eval.py \
        --config ${CONFIG} \
        --task ${TASK} \
        --samples ${SAMPLES} \
        --output-dir ${OUTPUT_DIR}
elif [ -n "${TASKS}" ]; then
    # Multiple tasks mode
    echo "Running multiple tasks: ${TASKS}"
    python /home/winnieyangwn/rllm/cookbooks/mlebench/eval_integration/eval.py \
        --config ${CONFIG} \
        --tasks ${TASKS} \
        --samples ${SAMPLES} \
        --output-dir ${OUTPUT_DIR}
else
    echo "ERROR: Must specify TASK or TASKS environment variable"
    echo "Usage: sbatch --export=TASK=mlsp-2013-birds launch.sh"
    exit 1
fi

EXIT_CODE=$?
echo "========================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
