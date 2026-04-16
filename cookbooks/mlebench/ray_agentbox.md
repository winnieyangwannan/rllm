# Ray + AgentBox Architecture for MLE-bench Evaluation

This document explains how Ray and AgentBox work together to run parallel MLE-bench evaluations.

## Overview

MLE-bench evaluation requires:
1. **Parallel rollout orchestration** - running many agent rollouts concurrently
2. **GPU sandboxes** - isolated containers with GPU access for running ML code
3. **LLM inference** - serving model requests for agent reasoning

These responsibilities are split across three systems:

| System | Responsibility | Resource Type |
|--------|---------------|---------------|
| **Ray** | Task scheduling, parallelism, retries | CPU workers |
| **AgentBox** | GPU container lifecycle management | GPU nodes |
| **vLLM** | LLM inference serving | GPU (separate) |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SLURM Job (eval job)                                  │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                         Ray Cluster (on this node)                        │  │
│  │                                                                           │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       ┌───────────┐  │  │
│  │   │ Ray Task 1  │  │ Ray Task 2  │  │ Ray Task 3  │  ...  │ Ray Task  │  │  │
│  │   │ (sample 0)  │  │ (sample 1)  │  │ (sample 2)  │       │ N         │  │  │
│  │   │ CPU worker  │  │ CPU worker  │  │ CPU worker  │       │ CPU wrkr  │  │  │
│  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       └─────┬─────┘  │  │
│  │          │                │                │                    │        │  │
│  └──────────┼────────────────┼────────────────┼────────────────────┼────────┘  │
└─────────────┼────────────────┼────────────────┼────────────────────┼────────────┘
              │                │                │                    │
              │  gRPC          │  gRPC          │  gRPC              │  gRPC
              ▼                ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         AgentBox Manager (central allocator)                    │
│                              SLURM Job (manager)                                │
└─────────────────────────────────────────────────────────────────────────────────┘
              │                │                │                    │
              │  Allocate      │  Allocate      │  Allocate          │  Allocate
              ▼                ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      AgentBox Workers (SLURM Job Array)                         │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Worker Node 1   │  │ Worker Node 2   │  │ Worker Node 3   │  ... (N nodes)  │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │                  │
│  │ │ Container 1 │ │  │ │ Container 2 │ │  │ │ Container 3 │ │                  │
│  │ │ GPU sandbox │ │  │ │ GPU sandbox │ │  │ │ GPU sandbox │ │                  │
│  │ │ (sample 0)  │ │  │ │ (sample 1)  │ │  │ │ (sample 2)  │ │                  │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │                  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────────┘
              │                │                │                    
              │  HTTP          │  HTTP          │  HTTP              
              ▼                ▼                ▼                    
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              vLLM Server (shared)                               │
│                           Model: e.g., Qwen3.5-27B                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### Ray Cluster

**Purpose**: Orchestrate parallel rollouts with explicit concurrency control.

**Key features**:
- Lightweight CPU tasks (no GPU needed)
- Sliding window concurrency via `ray.wait()`
- Unlimited retries for `WorkerDeadError` (transient worker failures)
- Incremental result saving

**Configuration** (in YAML):
```yaml
ray:
  max_concurrent_tasks_per_node: 32  # Tasks per node
```

**Retry behavior**:
- `WorkerDeadError` (worker died): Unlimited retries (`max_retries=-1`)
- Other exceptions: No retry, task fails immediately

**Code location**: `examples/mlebench/eval_ray.py`

### AgentBox Manager

**Purpose**: Central allocator that assigns containers to worker nodes.

**How it works**:
1. Ray Task requests a container via gRPC
2. Manager finds an available worker node
3. Manager tells worker to start container
4. Returns container handle to Ray Task

**Launch**: Separate SLURM job, runs on single node.

### AgentBox Workers

**Purpose**: GPU nodes that host sandboxed containers.

**Container features**:
- GPU access for ML training/inference
- Mounted data directory (`/root/data`)
- Isolated filesystem
- Apptainer/Singularity runtime

**Launch**: SLURM job array (e.g., 200 workers).

### vLLM Server

**Purpose**: Serve LLM inference requests from all containers.

**Shared resource**: All Ray Tasks hit the same vLLM instance.

**Configuration**:
```yaml
model:
  name: Qwen/Qwen3.5-27B
  backend: vllm
  base_url: http://h200-xxx-xxx-xxx:61558
```

## Flow: Single Rollout Lifecycle

```
1. Ray Task starts (sample_idx=5)
   └── Calls run_single_rollout()
   
2. run_single_rollout creates AgentBoxSandbox
   └── Connects to Manager (gRPC)
   └── Manager allocates a Worker node
   └── Worker starts Container (GPU sandbox)
   
3. Agent loop runs inside Ray Task
   └── Sends LLM requests → vLLM server (HTTP)
   └── Sends bash commands → Container via gRPC (shell.execute)
   
4. When rollout finishes
   └── Container closed, Worker freed
   └── Ray Task returns result
   └── Result saved to trajectories.jsonl
```

## Failure Handling: Worker Death

When a worker node dies (preempted, hardware failure), the container becomes unreachable.

### Before Fix (Infinite Retry Loop)

```
Container's worker dies
    ↓
gRPC error: "Connection refused"
    ↓
AgentBox retries connection (no limit)
    ↓
Stuck forever in retry loop ❌
```

### After Fix (WorkerDeadError + Ray Retry)

```
Container's worker dies
    ↓
gRPC error: "Connection refused"
    ↓
_safe_exec() retries 3 times (tools.py)
    ↓
WorkerDeadError raised after 3 failures
    ↓
eval.py catches error, re-raises for Ray
    ↓
Ray retries task on new worker (unlimited retries)
    ↓
Fresh container on healthy worker ✓
```

**Why unlimited retries is safe**: `WorkerDeadError` only occurs for transient infrastructure failures (worker preempted, hardware failure, network partition). These are NOT bugs in the rollout code - retrying on a different worker will eventually succeed. Other exceptions (OOM, model errors, bugs) are NOT `WorkerDeadError` and will fail immediately without retry.

### Timeline of Worker Death Recovery

```
TIME 0: Normal Operation
─────────────────────────
    Ray Task ──── gRPC ────► Worker h200-137-148-142 (ALIVE)
                             Container running ✓

TIME 1: Worker Dies
───────────────────
    Ray Task ──── gRPC ────► Worker h200-137-148-142 (DEAD 💀)
                             Connection refused!

TIME 2: Error Detection (~10-15s)
─────────────────────────────────
    _safe_exec() retries 3 times
    WorkerDeadError raised

TIME 3: Ray Catches Exception
─────────────────────────────
    Ray sees WorkerDeadError in retry_exceptions
    Ray schedules retry on different worker (unlimited retries)

TIME 4: Fresh Start
───────────────────
    Ray Task ──── gRPC ────► Worker h200-006-172 (HEALTHY)
                             NEW container allocated
                             Rollout starts fresh ✓
```

## Configuration Reference

### Eval Job YAML

```yaml
# Agent settings
agent:
  submit_file: code
  session_timeout: 1200      # 20 min per bash command
  eval_timeout: 86400        # 24 hours for train.py
  context_size: 262144       # Model context window
  temperature: 1.0
  llm_timeout: 900.0         # 15 min timeout per LLM call

# Sandbox settings
sandbox:
  manager_uri: h200-xxx-xxx-xxx:41921

# Ray settings  
ray:
  max_concurrent_tasks_per_node: 32
```

### Key Files

| File | Purpose |
|------|---------|
| `examples/mlebench/eval_ray.py` | Ray task orchestration |
| `examples/mlebench/eval.py` | Single rollout execution |
| `agenthub/mle_agent/mle_agent/tools.py` | Bash/tool execution with retry |
| `rllm/sdk/sandbox/backends/agentbox_backend.py` | AgentBox integration |

## Troubleshooting

### Check AgentBox Status

```bash
# Manager running?
squeue -j <manager_job_id>

# Workers running?
squeue -j <worker_job_array_id> --format="%.8T" | grep -c RUNNING

# Connection errors in eval logs?
grep -c "UNAVAILABLE" /path/to/logs/*.err
grep -c "Shell session failed" /path/to/logs/*.err
```

### Check vLLM Status

```bash
# Server metrics
curl http://<vllm_host>:61558/metrics | grep -E "num_requests|e2e_request"

# Requests running/waiting
curl -s http://<vllm_host>:61558/metrics | grep "num_requests_running\|num_requests_waiting"
```

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Shell session failed to start" | Worker node died | Wait for WorkerDeadError + retry |
| "Request timed out after 600.0s" | vLLM overloaded | Reduce concurrency or increase timeout |
| All tasks stuck | Manager down | Restart manager SLURM job |
| KV cache OOM | Too many concurrent requests | Reduce `max_concurrent_tasks_per_node` |
