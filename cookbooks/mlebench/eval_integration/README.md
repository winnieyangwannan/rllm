# MLE-bench Evaluation for rLLM

This directory contains the production evaluation script for MLE-bench with YAML configuration support.

## Quick Start

```bash
# Activate conda environment
conda activate rllm

# Test with single task (2 samples)
python eval.py --config configs/gpt5_test.yaml --task mlsp-2013-birds

# Production eval (64 samples)
python eval.py --config configs/gpt5.yaml --task mlsp-2013-birds

# List available tasks
python eval.py --config configs/base.yaml --list-tasks
```

## Directory Structure

```
eval_integration/
├── configs/
│   ├── base.yaml           # Shared defaults (inherited by others)
│   ├── gpt5.yaml           # GPT-5 production config (64 samples)
│   └── gpt5_test.yaml      # GPT-5 test config (2 samples)
├── eval.py                 # Main evaluation script
├── eval.md                 # Implementation plan
└── README.md               # This file
```

## Usage

### Single Task Evaluation

```bash
# Basic usage
python eval.py --config configs/gpt5_test.yaml --task mlsp-2013-birds

# Override number of samples
python eval.py --config configs/gpt5.yaml --task mlsp-2013-birds --samples 4

# Custom output directory
python eval.py --config configs/gpt5.yaml --task mlsp-2013-birds --output-dir /path/to/output

# Skip saving trajectories
python eval.py --config configs/gpt5.yaml --task mlsp-2013-birds --no-save
```

### Multiple Tasks

```bash
# Comma-separated list
python eval.py --config configs/gpt5.yaml --tasks mlsp-2013-birds,spooky-author-identification

# All tasks in JSONL directory
python eval.py --config configs/gpt5.yaml --all-tasks
```

### List Available Tasks

```bash
python eval.py --config configs/base.yaml --list-tasks
```

## Configuration

Configs use YAML with OmegaConf. Files can inherit from others using `defaults`:

```yaml
# gpt5.yaml
defaults:
  - base    # Inherits from base.yaml

model:
  name: gpt-5
  azure_endpoint: https://...
  api_key: ${oc.env:AZURE_API_KEY,default_key}
  api_version: 2025-03-01-preview

eval:
  samples_per_prompt: 64  # Overrides base.yaml
```

### Config Sections

| Section | Description |
|---------|-------------|
| `model` | LLM configuration (name, endpoint, credentials) |
| `agent` | Agent parameters (max_turns, timeouts, temperature) |
| `sandbox` | Container/sandbox settings (manager URI, images) |
| `data` | Data paths (MLE-bench data, JSONL directory) |
| `eval` | Evaluation settings (samples, output dir, parallelism) |

### Environment Variables

Sensitive values can use environment variables:

```yaml
api_key: ${oc.env:AZURE_API_KEY,fallback_value}
```

## Output

### Trajectory Files

Each rollout saves a JSON file to `output_dir`:

```
/checkpoint/.../RLLM/gpt5/
├── mlsp-2013-birds_0_20260413_143052.json
├── mlsp-2013-birds_1_20260413_143105.json
└── ...
```

Each file contains:
- Full Episode with Trajectory
- Conversation messages
- Config used
- Metrics (percentile, score, duration, steps)

### Console Output

```
TASK EVAL: mlsp-2013-birds
======================================================================
Config:
  model: gpt-5
  samples_per_prompt: 2
  ...

ROLLOUT 1: mlsp-2013-birds
============================================================
Data path: /checkpoint/.../mlebench/mlsp-2013-birds/prepared/public
✓ Created sandbox with data mount
✓ Agent completed with 45 steps in 1234.5s
✓ Evaluation complete:
  Percentile: 0.8750
  Signals: {'score': 0.95, 'valid_submission': 1.0}

EVALUATION SUMMARY
======================================================================
mlsp-2013-birds:
  Sample 1: ✓ percentile=0.8750, steps=45, duration=1234.5s
  Sample 2: ✓ percentile=0.9125, steps=52, duration=1456.2s
  Results: 2/2 successful
  Mean percentile: 0.8938
  Max percentile: 0.9125
```

## SLURM Integration

See Phase 1.5 in `eval.md` for SLURM launcher options:
- Option A: Simple `launch.sh` wrapper
- Option B: Python `launch.py` with dump directories

## Troubleshooting

### OmegaConf not found

```bash
conda activate rllm
pip install omegaconf
```

### AgentBox connection failed

Check that:
1. Manager URI is correct in config
2. You're on a node with access to the AgentBox cluster
3. Manager is running: `curl http://<manager_uri>/health`

### Task JSONL not found

Verify `data.task_jsonl_dir` in config points to directory containing `{task_id}.jsonl` files.

## Related Files

- `../initial_integration/test_step7_end_to_end.py` — Original standalone test
- `../../mle_agent/` — Agent implementation
- `eval.md` — Full implementation plan
