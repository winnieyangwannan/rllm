#!/bin/bash
#SBATCH --job-name=step9-new
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --partition=h200
#SBATCH --qos=h200_agentic-models_high
#SBATCH --account=agentic-models
#SBATCH --output=/checkpoint/maui_sft/winnieyangwn/rllm/step9_validation/logs/%j_new.out
#SBATCH --error=/checkpoint/maui_sft/winnieyangwn/rllm/step9_validation/logs/%j_new.err

# Phase 3 Step 9: Full validation with NEW agent loop (64 samples)

echo "========================================"
echo "Step 9 Validation: NEW Agent Loop"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Samples: 64"
echo "Manager URI: h200-137-001-039:45027"
echo "========================================"

# Setup conda
source /storage/home/winnieyangwn/miniforge3/etc/profile.d/conda.sh
conda activate rllm

# Clear any venv settings
unset VIRTUAL_ENV
unset PYTHONPATH

echo "Python: $(which python)"
echo "Python version: $(python --version)"

cd /home/winnieyangwn/rllm/examples/mlebench

# Output directory
OUTPUT_DIR="/checkpoint/maui_sft/winnieyangwn/rllm/step9_validation/new"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Running NEW agent loop with 64 samples..."
echo ""

python eval.py \
    --config configs/step9_new.yaml \
    --task mlsp-2013-birds \
    --samples 64 \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
