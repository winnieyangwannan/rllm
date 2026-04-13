# MLE-Bench Eval Plan for rLLM

This document traces the eval logic from amaia-collab and outlines a plan to create a similar eval script for MLE-bench in the rLLM codebase.

---

## Amaia-Collab Eval Flow Trace

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
   gen_args: {temperature, max_gen, ...}
   tasks:
     - env_config: mle_bench_bash
       reward_fn: mle_bench
       path: /.../mlebench_full.jsonl
       samples_per_prompt: 64
       init_args:
         config: {model, prompt_file, max_turns, ...}
   num_rollout_threads: 4
   ```

4. **Agent** (`apps/sea/agent.py`)
   - `Agent` class manages `Generator`, rollout threads, data queue, dump queue
   - `rollout_threads_start()` → parallel rollouts
   - Rollout strategies: `SimpleRollout`, `SelfRefineRollout`, etc.

5. **Args** (`apps/sea/args.py`)
   - `SeaRLEvalArgs(RLEvalArgs)`: litellm_args, metagen_args, tasks[], rollout_strategy
   - `AgentArgs`: generator config, tasks, num_rollout_threads, rollout_strategy

---

## Plan: MLE-Bench Eval Script for rLLM

### Option A: Extend the `rllm eval` CLI (Recommended)

**Why:** Follows rLLM's existing patterns — catalog-based discovery, unified runner, and eval decorators.

#### Files to Create/Modify:

1. **Register MLE-bench in dataset catalog** 
   - Add entry in `rllm/experimental/cli/templates/dataset_catalog.yaml`
   ```yaml
   mlebench:
     source: local  # or huggingface if published
     path: /checkpoint/maui_sft/winnieyangwn/datasets/mlebench_full.jsonl
     default_agent: mle_agent
     reward_fn: mlebench
     eval_split: test
     sandbox_required: true
   ```

2. **Register agent in agent catalog**
   - Add to `rllm/experimental/cli/templates/agent_catalog.yaml`
   ```yaml
   mle_agent:
     path: agenthub.mle_agent.mle_agent.agent:MLEAgentFlow
     description: Multi-turn bash agent for Kaggle competitions
     sandbox_backend: agentbox
   ```

3. **Register evaluator**
   - Either via `rllm/experimental/eval/evaluator_loader.py` or catalog

4. **Create YAML configs** (following amaia pattern)
   - Create `examples/mlebench/configs/` directory:
   ```
   examples/mlebench/
   ├── configs/
   │   ├── base.yaml           # Base config
   │   ├── gpt5.yaml           # GPT-5 specific
   │   └── qwen.yaml           # Qwen specific
   ├── eval.py                 # Entry point script
   └── README.md
   ```

5. **Config structure** (`examples/mlebench/configs/gpt5.yaml`):
   ```yaml
   # MLE-bench evaluation with GPT-5
   benchmark: mlebench
   model:
     name: azure/gpt-5
     base_url: https://azure-services-fair-openai1-eastus2n3.azure-api.net
     api_key: ${AZURE_API_KEY}
     api_version: 2025-03-01-preview
   
   agent:
     name: mle_agent
     max_turns: 128
     session_timeout: 43200  # 12 hours
     rollout_timeout: 86400  # 24 hours
     temperature: 1.0
     submit_file: csv
     check_submission_validity: true
   
   sandbox:
     backend: agentbox
     manager_uri: h200-137-001-093:42987
   
   eval:
     samples_per_prompt: 64
     num_workers: 4
     data_path: /checkpoint/maui_sft/winnieyangwn/datasets/mlebench_full.jsonl
     dump_dir: /checkpoint/agentic-models/winnieyangwn/amaia_dumps
   ```

6. **Entry script** (`examples/mlebench/eval.py`):
   ```python
   """MLE-bench evaluation script.
   
   Usage:
       python examples/mlebench/eval.py --config configs/gpt5.yaml
       python examples/mlebench/eval.py --config configs/gpt5.yaml --task mlsp-2013-birds
   """
   from omegaconf import OmegaConf
   from rllm.experimental.eval.runner import EvalRunner
   # ... load config, run evaluation
   ```

### Option B: Standalone Script (Current test_end_to_end.py approach)

Keep and enhance `cookbooks/mlebench/test_end_to_end.py` but add YAML config support.

---

## Implementation Steps

| Step | Action | Files |
|------|--------|-------|
| 1 | Add mlebench to dataset catalog | `rllm/experimental/cli/templates/dataset_catalog.yaml` |
| 2 | Add mle_agent to agent catalog | `rllm/experimental/cli/templates/agent_catalog.yaml` |  
| 3 | Update MLEAgentFlow to load config from AgentConfig.metadata | `agenthub/mle_agent/mle_agent/agent.py` |
| 4 | Create config directory structure | `examples/mlebench/configs/` |
| 5 | Create base.yaml + model-specific configs | `examples/mlebench/configs/*.yaml` |
| 6 | Create eval.py entry script with YAML loading | `examples/mlebench/eval.py` |
| 7 | Add SLURM launcher (optional, for parity with amaia) | `examples/mlebench/launch.py` |

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

## Existing rLLM Components to Leverage

- **MLEAgentFlow**: `agenthub/mle_agent/mle_agent/agent.py` - already implements agent loop
- **MLEEvaluator**: `agenthub/mle_agent/mle_agent/evaluator.py` - 7-stage grading pipeline
- **SandboxedAgentFlow**: `rllm/experimental/agents/sandboxed_agent.py` - base class for sandbox lifecycle
- **EvalRunner**: `rllm/experimental/eval/runner.py` - parallel evaluation orchestration
- **AgentConfig**: `rllm/experimental/eval/types.py` - config injection into agents

---

## Next Steps

1. Decide between Option A (full catalog integration) vs Option B (standalone with YAML)
2. Create directory structure under `examples/mlebench/`
3. Implement YAML config loading in eval.py
4. Test with single task before scaling to full benchmark
