# MLE-Bench Eval Plan for rLLM

This document traces the eval logic from amaia-collab (/home/winnieyangwn/amaia-collab) and outlines a plan to create a similar eval script for MLE-bench in the rLLM codebase.

---

## What's Already Working (Tested ✅)

The following tests in `cookbooks/mlebench/` have all passed:

| Step | Test File | What It Proves |
|------|-----------|----------------|
| 1 | `test_step1_agentbox_connection.py` | AgentBox manager connection, container lifecycle |
| 2 | `test_step2_agentbox_sandbox.py` | `AgentBoxSandbox` wrapper (exec, file ops) |
| 2b | `test_step2b_data_real.py` | Task data loading from JSONL |
| 3 | `test_step3_prompts.py` | System/instance prompt formatting |
| 4 | `test_step4_agent_loop.py` | `_run_agent_loop()` with tool calling |
| 5 | `test_step5_agent_real.py` | Full agent with real LLM |
| 6 | `test_step6_evaluator.py` | `MLEEvaluator` 7-stage pipeline |
| 7 | `test_step7_end_to_end.py` | **Complete flow**: task → agent → eval → percentile |

**Key proven components:**
- `MLEAgentFlow` in `agenthub/mle_agent/mle_agent/agent.py`
- `MLEEvaluator` in `agenthub/mle_agent/mle_agent/evaluator.py`
- `AgentBoxSandbox` in `rllm/sdk/sandbox/backends/agentbox_backend.py`

---

## Amaia-Collab Eval Flow Trace (Reference)

**Command:**
```bash
python -m launchers.stool run \
    name="526_sample_64" \
    script=apps.sea.eval \
    config=apps/sea/configs/winnieyang/eval/baseline/gpt5/526.yaml \
    nodes=3 \
    group=agentic-models \
    qos=h200_agentic-models_high \
    dirs_exists_ok=True
```

### Flow:

1. **Launcher** (`launchers/stool/cli.py`)
   - Parses `StoolRunArgs`: `name`, `script`, `config`, `nodes`, `qos`, etc.
   - Creates dump directory at `{root_dump_dir}/{name}`
   - Copies code to `{dump_dir}/code/{launch_date_str}`
   - Submits SLURM job via `sbatch`

2. **Eval Entry** (`apps/sea/eval.py`)
   - `main()` → `load_from_cli(SeaRLEvalArgs)` loads YAML config
   - `eval_agent(args)` creates dump directories, runs `run_agent_evals()`
   
3. **Config** (`526.yaml`)
   ```yaml
   dump_dir: /checkpoint/.../526
   gen_backend: litellm
   litellm_args: {model, api_key, base_url, tools_env, ...}
   tasks:
     - env_config: mle_bench_bash
       reward_fn: mle_bench
       path: /.../mlebench_full.jsonl
       samples_per_prompt: 64
   num_rollout_threads: 4
   ```

---

## Plan: MLE-Bench Eval Script for rLLM

### Recommended: Hybrid Approach

Use the working `test_step7_end_to_end.py` as base, add YAML config support, and optionally integrate with CLI later.

**Why:** The existing loaders already support explicit import paths, so we can skip catalog integration initially.

---

## Key Discoveries: How to Fix Original Problems

### Problem 1-2: Registry Location & Format ✅ EASY

**Original plan was wrong.** The actual files are:
- `rllm/registry/datasets.json` (JSON, not YAML)
- `rllm/registry/agents.json` (JSON, not YAML)

**Fix:** Add JSON entries to the correct files.

### Problem 3: Local Dataset Loading ⚠️ NEEDS WORKAROUND

Current `DatasetRegistry` only supports HuggingFace. Options:

**Option A (Recommended): Bypass catalog, load directly**
```python
# In eval.py - load tasks directly like test_step7_end_to_end.py does
def load_tasks_from_jsonl(jsonl_path: str) -> list[dict]:
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]
```

**Option B: Add `source: jsonl` support** (future work)
- Modify `rllm/experimental/cli/_pull.py` to handle local files
- More invasive, defer to later phase

### Problem 4: Agent Loading ✅ ALREADY WORKS

The loader supports explicit import paths:
```python
# In agent_loader.py line 118
if ":" in name_or_path:
    return _load_and_instantiate(name_or_path, name_or_path)
```

**Usage:**
```bash
rllm eval ... --agent "agenthub.mle_agent.mle_agent.agent:mle_agent"
```

Or add to `rllm/registry/agents.json`:
```json
"mle_agent": {
  "description": "Multi-turn bash agent for Kaggle competitions",
  "module": "agenthub.mle_agent.mle_agent.agent",
  "function": "mle_agent"
}
```

### Problem 5: Evaluator Loading ✅ ALREADY WORKS

Same pattern as agents:
```bash
rllm eval ... --evaluator "agenthub.mle_agent.mle_agent.evaluator:MLEEvaluator"
```

Or add to `_EVALUATOR_REGISTRY` in `evaluator_loader.py`:
```python
"mlebench_reward_fn": MLEEvaluator,  # after importing
```

### Problem 6: Config Injection ⚠️ NEEDS SMALL FIX

`MLEAgentFlow.__init__()` takes params like `manager_uri`, `data_base_path`, but:
- The module-level singleton `mle_agent = MLEAgentFlow()` has empty defaults
- `EvalRunner` passes config via `AgentConfig.metadata`

**Fix:** Update `MLEAgentFlow.setup_sandbox()` to read from `config.metadata`:
```python
def setup_sandbox(self, task: dict, config) -> None:
    # Fallback to metadata if instance attrs are empty
    manager_uri = self.manager_uri or config.metadata.get("manager_uri", "")
    data_base_path = self.data_base_path or config.metadata.get("data_base_path", "")
    # ... rest of setup
```

---

## Revised Implementation Plan

### Phase 1: Standalone Eval Script (Leverage Working Code)

| Step | Action | Effort |
|------|--------|--------|
| 1.1 | Copy `test_step7_end_to_end.py` to `examples/mlebench/eval.py` | 5 min |
| 1.2 | Add YAML config loading (OmegaConf) | 30 min |
| 1.3 | Externalize hardcoded values to config | 30 min |
| 1.4 | Add trajectory saving to output dir | 20 min |
| 1.5 | Test with single task | 10 min |

**Deliverable:** `python examples/mlebench/eval.py --config configs/gpt5.yaml --task mlsp-2013-birds`

### Phase 1.5: SLURM Launcher Support

Two options for cluster job submission:

#### Option A: Simple Shell Script (Quick Start)

| Step | Action | Effort |
|------|--------|--------|
| 1.5.1a | Create `launch.sh` with SBATCH headers | 15 min |
| 1.5.2a | Test: `sbatch --export=TASK=mlsp-2013-birds launch.sh` | 5 min |

`examples/mlebench/launch.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=mlebench-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=h200_agentic-models
#SBATCH --qos=h200_agentic-models_high
#SBATCH --output=/checkpoint/agentic-models/%u/slurm_logs/%j.out

source /home/winnieyangwn/miniconda3/etc/profile.d/conda.sh
conda activate rllm
cd /home/winnieyangwn/rllm

python examples/mlebench/eval.py \
    --config examples/mlebench/configs/gpt5.yaml \
    --task ${TASK:-mlsp-2013-birds} \
    --samples ${SAMPLES:-64}
```

**Usage:**
```bash
sbatch --export=TASK=mlsp-2013-birds,SAMPLES=64 examples/mlebench/launch.sh
```

#### Option B: Python Launcher (Reproducibility Features)

| Step | Action | Effort |
|------|--------|--------|
| 1.5.1b | Create `launch.py` with dump dir management | 1.5 hrs |
| 1.5.2b | Add code copying for reproducibility | 30 min |
| 1.5.3b | Test: `python launch.py --config configs/gpt5.yaml --name exp_001` | 10 min |

`examples/mlebench/launch.py`:
```python
"""SLURM launcher for MLE-bench evaluation.

Usage:
    python examples/mlebench/launch.py \
        --config configs/gpt5.yaml \
        --name my_experiment \
        --nodes 1 \
        --tasks mlsp-2013-birds,spooky-author-identification
"""
import argparse
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--tasks", help="Comma-separated task IDs or 'all'")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--partition", default="h200_agentic-models")
    parser.add_argument("--qos", default="h200_agentic-models_high")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--dump-dir", default="/checkpoint/agentic-models/winnieyangwn/mlebench_dumps")
    args = parser.parse_args()

    # Create dump directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_dir = Path(args.dump_dir) / f"{args.name}_{timestamp}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy code for reproducibility
    code_dir = dump_dir / "code"
    shutil.copytree("/home/winnieyangwn/rllm", code_dir, 
                    ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '.venv'))
    
    # Copy config
    shutil.copy(args.config, dump_dir / "config.yaml")
    
    # Generate sbatch script
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={args.name}
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time={args.time}
#SBATCH --partition={args.partition}
#SBATCH --qos={args.qos}
#SBATCH --output={dump_dir}/logs/%j.out
#SBATCH --error={dump_dir}/logs/%j.err

source /home/winnieyangwn/miniconda3/etc/profile.d/conda.sh
conda activate rllm
cd {code_dir}

python examples/mlebench/eval.py \\
    --config {dump_dir}/config.yaml \\
    --output-dir {dump_dir}/results \\
    --samples {args.samples} \\
    {"--tasks " + args.tasks if args.tasks else ""}
"""
    
    # Write and submit
    (dump_dir / "logs").mkdir(exist_ok=True)
    sbatch_path = dump_dir / "run.sbatch"
    sbatch_path.write_text(sbatch_script)
    
    print(f"Dump directory: {dump_dir}")
    print(f"Submitting job...")
    
    result = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python examples/mlebench/launch.py \
    --config configs/gpt5.yaml \
    --name exp_526 \
    --tasks mlsp-2013-birds,spooky-author-identification \
    --samples 64
```

**Features:**
- Creates timestamped dump directory
- Copies code for reproducibility
- Auto-generates sbatch script
- Captures logs in dump directory

**Deliverable:** SLURM job submission with either quick script or full launcher

### Phase 2: Registry Integration (Optional CLI Support)

| Step | Action | Effort |
|------|--------|--------|
| 2.1 | Add `mle_agent` to `rllm/registry/agents.json` | 5 min |
| 2.2 | Add `mlebench_reward_fn` to `evaluator_loader.py` registry | 10 min |
| 2.3 | Update `MLEAgentFlow` to read from `config.metadata` | 30 min |
| 2.4 | Test: `rllm eval --agent mle_agent --evaluator mlebench_reward_fn ...` | 15 min |

**Deliverable:** CLI integration (partial - still needs custom data loading)

### Phase 3: Full CLI Integration (Future)

| Step | Action | Effort |
|------|--------|--------|
| 3.1 | Add `source: jsonl` support to dataset loader | 2 hrs |
| 3.2 | Add `mlebench` to `rllm/registry/datasets.json` | 10 min |
| 3.3 | Test: `rllm eval mlebench --model gpt-5` | 30 min |

---

## Config Structure (Phase 1)

`examples/mlebench/configs/gpt5.yaml`:
```yaml
# MLE-bench evaluation with GPT-5
model:
  name: gpt-5
  azure_endpoint: https://azure-services-fair-openai1-eastus2n3.azure-api.net
  api_key: ${AZURE_API_KEY}
  api_version: 2025-03-01-preview

agent:
  max_turns: 128
  session_timeout: 1200      # 20 min per bash call
  rollout_timeout: 43200     # 12 hours total
  context_size: 98304
  temperature: 1.0
  submit_file: csv
  check_submission_validity: true

sandbox:
  manager_uri: h200-137-000-067:42499
  superimage_directory: /checkpoint/maui_sft/shared/sif
  superimage_version: 2025-05-02v2
  superimage_overlay: /checkpoint/fair-maui-hs/hotfix/kniu.2025-09-19.cache.overlay.ext3.img

data:
  mle_bench_data_dir: /checkpoint/maui/shared/cache/dojo/tasks/mlebench
  task_jsonl_dir: /checkpoint/maui_sft/winnieyangwn/datasets

eval:
  samples_per_prompt: 64
  output_dir: /checkpoint/maui_sft/winnieyangwn/RLLM
```

---

## Remaining Open Questions

1. **HF Dataset Publishing?** — Could upload task JSONL to HuggingFace to enable standard `rllm eval mlebench`. Lower priority.

2. **Multi-node parallelism?** — Current SLURM launcher (Phase 1.5) runs on single node. For true multi-node, would need job arrays or Ray/MPI coordination. Consider for Phase 4 if needed.

---

## Key Convention Differences

| Aspect | amaia-collab | rLLM |
|--------|--------------|------|
| Config loading | `load_from_cli(Args)` + YAML | OmegaConf or dataclass |
| Launcher | `stool` → sbatch | Direct CLI or scripts |
| Agent | `Agent` + `Generator` + rollout threads | `AgentFlow` + `EvalRunner` |
| Parallelism | `num_rollout_threads` + queues | `concurrency` semaphore + asyncio |
| Eval | Integrated in agent | Separate `Evaluator` protocol |

---

## Files to Create/Modify

### Phase 1 (New Files)
```
examples/mlebench/
├── configs/
│   ├── base.yaml           # Shared defaults
│   └── gpt5.yaml           # GPT-5 specific
├── eval.py                 # Entry point (based on test_step7_end_to_end.py)
└── README.md
```

### Phase 1.5 (SLURM Launcher Files)
```
examples/mlebench/
├── launch.sh               # Option A: Simple sbatch wrapper
└── launch.py               # Option B: Python launcher with dump dirs
```

### Phase 2 (Modifications)
- `rllm/registry/agents.json` — add `mle_agent` entry
- `rllm/experimental/eval/evaluator_loader.py` — add `mlebench_reward_fn`
- `agenthub/mle_agent/mle_agent/agent.py` — read config from metadata

### Phase 3 (Modifications)
- `rllm/registry/datasets.json` — add `mlebench` entry
- `rllm/experimental/cli/_pull.py` — support `source: jsonl`
