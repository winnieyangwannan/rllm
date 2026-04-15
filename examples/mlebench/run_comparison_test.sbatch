#!/bin/bash
#SBATCH --job-name=comparison-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --partition=h200
#SBATCH --qos=h200_agentic-models_high
#SBATCH --account=agentic-models
#SBATCH --output=/checkpoint/maui_sft/winnieyangwn/rllm/comparison_test/logs/%j.out
#SBATCH --error=/checkpoint/maui_sft/winnieyangwn/rllm/comparison_test/logs/%j.err

# Phase 3: Side-by-side comparison test
# Runs mlsp-2013-birds with both legacy and new agent loop

echo "========================================"
echo "Phase 3: Side-by-Side Comparison Test"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
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

# Create output directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="/checkpoint/maui_sft/winnieyangwn/rllm/comparison_test/${TIMESTAMP}"
LEGACY_OUTPUT="${OUTPUT_BASE}/legacy"
NEW_OUTPUT="${OUTPUT_BASE}/new"

mkdir -p "$LEGACY_OUTPUT"
mkdir -p "$NEW_OUTPUT"

echo ""
echo "========================================"
echo "1/2: Running LEGACY agent loop"
echo "========================================"
echo ""

python eval.py \
    --config configs/comparison_test.yaml \
    --task mlsp-2013-birds \
    --samples 1 \
    --output-dir "$LEGACY_OUTPUT" \
    --legacy-agent-loop

LEGACY_EXIT=$?
echo ""
echo "Legacy exit code: $LEGACY_EXIT"

echo ""
echo "========================================"
echo "2/2: Running NEW agent loop"
echo "========================================"
echo ""

python eval.py \
    --config configs/comparison_test.yaml \
    --task mlsp-2013-birds \
    --samples 1 \
    --output-dir "$NEW_OUTPUT"

NEW_EXIT=$?
echo ""
echo "New exit code: $NEW_EXIT"

echo ""
echo "========================================"
echo "Results Summary"
echo "========================================"
echo ""
echo "Legacy results: $LEGACY_OUTPUT/mlsp-2013-birds.jsonl"
echo "New results:    $NEW_OUTPUT/mlsp-2013-birds.jsonl"
echo ""

# Quick comparison
if [ -f "$LEGACY_OUTPUT/mlsp-2013-birds.jsonl" ] && [ -f "$NEW_OUTPUT/mlsp-2013-birds.jsonl" ]; then
    echo "Legacy file size: $(wc -c < "$LEGACY_OUTPUT/mlsp-2013-birds.jsonl") bytes"
    echo "New file size:    $(wc -c < "$NEW_OUTPUT/mlsp-2013-birds.jsonl") bytes"
    
    # Compare termination reason
    echo ""
    echo "Legacy termination reason:"
    grep -o '"termination_reason":"[^"]*"' "$LEGACY_OUTPUT/mlsp-2013-birds.jsonl" | head -1
    echo "New termination reason:"
    grep -o '"termination_reason":"[^"]*"' "$NEW_OUTPUT/mlsp-2013-birds.jsonl" | head -1
    
    # Compare if both submitted
    echo ""
    LEGACY_SUBMITTED=$(grep -c '"pred_solution":' "$LEGACY_OUTPUT/mlsp-2013-birds.jsonl" || echo 0)
    NEW_SUBMITTED=$(grep -c '"pred_solution":' "$NEW_OUTPUT/mlsp-2013-birds.jsonl" || echo 0)
    
    if [ "$LEGACY_SUBMITTED" -gt 0 ] && [ "$NEW_SUBMITTED" -gt 0 ]; then
        echo "✓ Both runs submitted solutions"
    elif [ "$LEGACY_SUBMITTED" -eq 0 ] && [ "$NEW_SUBMITTED" -eq 0 ]; then
        echo "✓ Neither run submitted (consistent)"
    else
        echo "⚠ Mismatch: legacy submitted=$LEGACY_SUBMITTED, new submitted=$NEW_SUBMITTED"
    fi
fi

echo ""
echo "========================================"
echo "End time: $(date)"
echo "========================================"

# Return 0 if both succeeded
if [ $LEGACY_EXIT -eq 0 ] && [ $NEW_EXIT -eq 0 ]; then
    echo "✓ COMPARISON COMPLETED SUCCESSFULLY"
    exit 0
else
    echo "⚠ One or both runs failed"
    exit 1
fi
