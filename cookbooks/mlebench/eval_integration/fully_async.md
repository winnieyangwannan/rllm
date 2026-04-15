# rLLM Fully Async Training Architecture

Notes on how fully async training works in rllm for future MLE-bench integration.

## Overview

Fully async mode **decouples inference (rollout) from training** — they run concurrently as separate Ray actors connected via an async MessageQueue.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FullyAsyncTaskRunner (orchestrator)                   │
│                              @ray.remote(num_cpus=1)                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
       ┌───────────────────────────────┼───────────────────────────────┐
       ▼                               ▼                               ▼
┌─────────────────┐  trajectories  ┌──────────────┐  gradients   ┌──────────────────┐
│ RolloutExecutor │───────────────▶│ MessageQueue │─────────────▶│ FullyAsyncTrainer│
│ @ray.remote     │                │ @ray.remote  │              │ @ray.remote      │
│ (num_cpus=10)   │                │ (num_cpus=2) │              │ (num_cpus=10)    │
└────────┬────────┘                └──────────────┘              └────────┬─────────┘
         │ HTTP                                                           │
         ▼                                                                │
┌─────────────────┐     pause/resume/sync_weights     ┌───────────────────┴────────┐
│InferenceManager │◀──────────────────────────────────│  ParameterSynchronizer     │
│ (SGLang servers)│              NCCL                 │  @ray.remote               │
│ @ray.remote     │◀══════════════════════════════════│  (broadcast actor weights) │
│ (num_cpus=10)   │                                   └────────────────────────────┘
└─────────────────┘
```

---

## Key Components

### 1. FullyAsyncTaskRunner — Orchestrator
**File:** `rllm/experimental/fully_async/runner.py`
**Ray Actor:** `@ray.remote(num_cpus=1)`

Main entry point that initializes all components and runs the training loop:

```python
def _initialize_components(self, config):
    # 1. Create InferenceManager (SGLang servers)
    self.inference_manager = InferenceManager.remote(...)
    ray.get(self.inference_manager.init_workers.remote())
    self.router_url = ray.get(self.inference_manager.launch_router.remote())
    
    # 2. Create RolloutExecutor (owns dataset, staleness logic)
    self.rollout_executor = RolloutExecutor.remote(router_url=self.router_url, ...)
    
    # 3. Create MessageQueue for trajectory buffering
    self.message_queue = MessageQueue.remote(config, max_queue_size)
    
    # 4. Create FullyAsyncTrainer
    self.trainer = FullyAsyncTrainer.remote(...)
    
    # 5. Create ParameterSynchronizer
    self.param_synchronizer = ParameterSynchronizer.remote(...)
    
    # 6. Initial weight sync
    ray.get(self.param_synchronizer.sync_weights.remote(version=0))

def _run_training_loop(self):
    # Both run in parallel — Ray handles scheduling
    rollout_future = self.rollout_executor.fit.remote()
    trainer_future = self.trainer.fit.remote()
    ray.wait([rollout_future, trainer_future])
```

**Factory function to inject custom rollout:**
```python
from rllm.experimental.fully_async.runner import create_task_runner_with_rollout_fn

ConfiguredRunner = create_task_runner_with_rollout_fn(my_rollout_fn)
runner = ConfiguredRunner.remote()
ray.get(runner.run.remote(config))
```

---

### 2. RolloutExecutor — Inference Node
**File:** `rllm/experimental/fully_async/rollout_executor.py`
**Ray Actor:** `@ray.remote(num_cpus=10, max_concurrency=10)`

Generates trajectories concurrently via HTTP to SGLang servers:

```python
class RolloutExecutor:
    def __init__(self, router_url, rollout_fn, n, config, ...):
        self.client = RolloutClient(router_url=router_url, ...)
        self.dataloader = StatefulDataLoader(dataset, batch_size=1, shuffle=True)
        
        # Staleness control
        self.max_staleness_samples = required_samples * (staleness_threshold + 1) * sync_step
        
        # Concurrency control
        self.sema = asyncio.Semaphore(128 * num_servers)  # Max concurrent rollouts
        self.continue_event = asyncio.Event()  # Throttle when queue full

    async def fit(self):
        for datum in self.dataloader:
            # Throttle if queue full (staleness control)
            if self.active + self.enqueued >= self.max_staleness_samples:
                self.continue_event.clear()
            
            await self.continue_event.wait()
            await self.sema.acquire()
            asyncio.create_task(self.generate_trajectory(datum))
    
    async def generate_trajectory(self, datum):
        result = await self.rollout_fn(self.client, self.tokenizer, **datum)
        
        if len(self.result_dict[idx]) >= self.n:  # n rollouts per prompt
            group = TrajectoryGroup(trajectories=[...])
            await self.trajectory_queue.put(cloudpickle.dumps(group))
```

**Rollout function signature:**
```python
async def rollout_fn(client: RolloutClient, tokenizer, **kwargs) -> Trajectory:
    """
    Args:
        client: Async HTTP client for SGLang with pause/resume support
        tokenizer: For encoding/decoding
        **kwargs: Task data fields (instance_id, task_description, etc.)
    Returns:
        Trajectory with steps, output, reward
    """
```

---

### 3. RolloutClient — Async HTTP Client
**File:** `rllm/experimental/fully_async/client.py`

Direct HTTP client for SGLang with weight sync support:

```python
class RolloutClient:
    def __init__(self, router_url, tokenizer, max_concurrency=4096):
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency),
            timeout=httpx.Timeout(None),
        )
        self.cur_version = 0       # Current param version
        self.resume_event = asyncio.Event()  # Pause/resume for weight sync
    
    async def generate(self, prompt_ids, sampling_params) -> OutputWithVersion:
        """Low-level: generate with token IDs."""
        while True:
            await self.resume_event.wait()  # Block if paused for weight sync
            output = await self._generate(prompt_ids, sampling_params)
            if output.finish_reason == "abort":  # Weight sync interrupted
                continue  # Retry with new weights
            return output
    
    async def chat_completion(self, messages, sampling_params, tools=None):
        """High-level: chat completion with tool calling."""
        prompt_ids = self.tokenizer.apply_chat_template(messages, tools=tools)
        output = await self.generate(prompt_ids, sampling_params)
        message = parse_response(self.tokenizer, output)
        return message, output

    def pause(self):
        """Called during weight sync."""
        self.resume_event.clear()
    
    def resume(self):
        """Called after weight sync."""
        self.resume_event.set()
```

**Key difference from OpenAI client:**
| Feature | OpenAI Client | RolloutClient |
|---------|--------------|---------------|
| Interface | `client.chat.completions.create()` | `await client.chat_completion()` |
| Sync/Async | Sync | Async only |
| Weight tracking | None | `cur_version` |
| Pause/resume | No | Yes (`resume_event`) |
| Return | `ChatCompletion` | `(message, OutputWithVersion)` |

---

### 4. MessageQueue — Trajectory Buffer
**File:** `rllm/experimental/fully_async/message_queue.py`
**Ray Actor:** `@ray.remote(num_cpus=2, max_concurrency=20)`

Bounded async queue connecting rollout to training:

```python
class MessageQueue:
    def __init__(self, config, max_queue_size):
        self.queue = deque(maxlen=max_queue_size)
        self._consumer_condition = asyncio.Condition()
    
    async def put_sample(self, sample) -> bool:
        async with self._lock:
            if len(self.queue) >= self.max_queue_size:
                self.queue.popleft()  # Drop oldest (FIFO)
                self.dropped_samples += 1
            self.queue.append(sample)
            self._consumer_condition.notify_all()
    
    async def get_sample(self):
        async with self._lock:
            while len(self.queue) == 0:
                await self._consumer_condition.wait()  # Block until available
            return self.queue.popleft(), len(self.queue)
```

---

### 5. FullyAsyncTrainer — Training Node
**File:** `rllm/experimental/fully_async/fully_async_trainer.py`
**Ray Actor:** `@ray.remote(num_cpus=10)`

Consumes trajectories and performs training:

```python
class FullyAsyncTrainer:
    async def fit(self):
        while True:
            # 1. Get batch from queue (blocks until required_samples)
            batch = self._get_samples_from_queue()
            
            # 2. Compute log probs, values, advantages
            batch = self._process_batch_common(batch)
            
            # 3. Update actor and critic
            self.actor_rollout_wg.update_actor(batch)
            
            # 4. Trigger weight sync every N steps
            if self.global_steps % trigger_sync_step == 0:
                await self._trigger_parameter_sync()
            
            self.global_steps += 1
```

---

### 6. ParameterSynchronizer — Weight Sync
**File:** `rllm/experimental/fully_async/param_sync.py`
**Ray Actor:** `@ray.remote`

Coordinates weight updates between trainer and inference:

```python
class ParameterSynchronizer:
    def __init__(self, trainer, inference_manager, ...):
        # NCCL collective group for efficient GPU-to-GPU broadcast
        collective.create_collective_group(
            actor_rollout_workers, backend="nccl", group_name="actor_rollout"
        )
    
    def sync_weights(self, version):
        # 1. Pause rollout generation
        ray.get(self.rollout_executor.pause.remote())  # Aborts in-flight requests
        
        # 2. Clear KV cache (release GPU memory)
        ray.get(self.inference_manager.clear_kv_cache.remote())
        
        # 3. Update param version
        ray.get(self.rollout_executor.update_param_version.remote(version))
        
        # 4. NCCL broadcast: trainer weights → inference weights
        self.actor_wg.sync_rollout_weights(group_name)
        ray.get(self.rollout_wg.sync_rollout_weights(group_name))
        
        # 5. Resume rollout (retries aborted requests with new weights)
        ray.get(self.rollout_executor.resume.remote())
```

**Weight sync sequence:**
```
Trainer               ParameterSynchronizer          RolloutExecutor         InferenceManager
   │                         │                            │                        │
   │ sync_weights(v=1)       │                            │                        │
   │────────────────────────▶│                            │                        │
   │                         │ pause()                    │                        │
   │                         │───────────────────────────▶│                        │
   │                         │                            │ client.pause()         │
   │                         │                            │ abort in-flight        │
   │                         │ clear_kv_cache()           │                        │
   │                         │────────────────────────────┼───────────────────────▶│
   │                         │                            │                        │
   │                         │══════ NCCL broadcast weights ═════════════════════▶│
   │                         │                            │                        │
   │                         │ resume()                   │                        │
   │                         │───────────────────────────▶│                        │
   │                         │                            │ client.resume()        │
   │◀────────────────────────│                            │                        │
```

---

## Data Flow: End-to-End

```
1. Task → RolloutExecutor
   └── StatefulDataLoader yields task dicts

2. RolloutExecutor → RolloutClient → SGLang
   └── await client.chat_completion(messages, sampling_params, tools)
   └── HTTP to SGLang router → one of N inference servers
   └── Returns (message, OutputWithVersion) with prompt_ids, response_ids, logprobs

3. Build Trajectory
   └── Trajectory(steps=[...], output=answer, metadata={"param_version": client.cur_version})

4. Group n rollouts per prompt
   └── TrajectoryGroup(trajectories=[t1, t2, ...])
   └── await trajectory_queue.put(cloudpickle.dumps(group))

5. MessageQueue buffers
   └── Bounded deque, drops oldest if full

6. FullyAsyncTrainer consumes
   └── sample = message_queue.get_sample()  # blocks if empty
   └── batch = assemble_batch_from_trajectory_groups([...])

7. Training step
   └── compute_log_prob(batch) → compute_values(batch) → compute_advantages(batch)
   └── update_critic(batch) → update_actor(batch)

8. Weight sync (every N steps)
   └── ParameterSynchronizer.sync_weights()
   └── pause rollout → clear KV cache → NCCL broadcast → resume rollout
```

---

## Staleness Management

**Config:**
```yaml
async_training:
  staleness_threshold: 0.1           # Allow 10% extra samples
  trigger_parameter_sync_step: 4     # Sync every 4 training steps
  required_samples: 128              # Samples per training step
```

**Queue size = 128 × 1.1 × 4 = 563 samples max**

**Throttling:**
- If `active + enqueued >= max_staleness_samples`: pause generation
- Weight sync resets quota: `enqueued = actual_queue_size`

**Version tracking:**
```python
# Each trajectory records which param version generated it
trajectory.metadata["param_version"] = client.cur_version

# Trainer can compute staleness metrics
batch.meta_info["rollout_param_versions"] = [...]
```

---

## Ray Actors Summary

| Component | Decorator | Resources | Role |
|-----------|-----------|-----------|------|
| `FullyAsyncTaskRunner` | `@ray.remote(num_cpus=1)` | 1 CPU | Orchestrator |
| `RolloutExecutor` | `@ray.remote(num_cpus=10, max_concurrency=10)` | 10 CPUs | Async rollout generation |
| `FullyAsyncTrainer` | `@ray.remote(num_cpus=10)` | 10 CPUs | Training loop |
| `MessageQueue` | `@ray.remote(num_cpus=2, max_concurrency=20)` | 2 CPUs | Trajectory buffer |
| `InferenceManager` | `@ray.remote(num_cpus=10, max_concurrency=100)` | 10 CPUs | SGLang servers |
| `ParameterSynchronizer` | `@ray.remote` | Default | Weight sync coordination |

---

## MLE-bench Integration for Fully Async

For MLE-bench to work with fully async training:

### 1. Async Agent Loop

Current `_run_agent_loop()` uses sync OpenAI client. Need async version:

```python
# mle_agent/agent_async.py

async def _run_agent_loop_async(
    client: RolloutClient,
    tokenizer,
    messages: list[dict],
    sandbox: Sandbox,
    **kwargs
) -> tuple[list[Step], list[dict], str | None, dict]:
    """Async agent loop using RolloutClient."""
    
    while not done:
        # RolloutClient handles pause/resume during weight sync
        message, output = await client.chat_completion(
            messages, sampling_params, tools=tools
        )
        
        # Track weight version
        step.metadata["param_version"] = client.cur_version
        
        # ... rest of agent loop (tool execution, etc.)
```

### 2. Rollout Function

```python
# train_integration/rollout_async.py

async def mle_bench_rollout_fn(client: RolloutClient, tokenizer, **kwargs) -> Trajectory:
    """Rollout function for FullyAsyncTaskRunner."""
    
    task_data = kwargs
    sandbox = create_sandbox(task_data, cfg)
    messages = build_initial_messages(task_data, cfg)
    
    steps, final_messages, pred_solution, metrics = await _run_agent_loop_async(
        client=client,
        tokenizer=tokenizer,
        messages=messages,
        sandbox=sandbox,
        **agent_kwargs
    )
    
    return Trajectory(
        steps=steps,
        output=pred_solution,
        metadata={"param_version": client.cur_version},
    )
```

### 3. Training Script

```python
# train_integration/train_async.py

from rllm.experimental.fully_async.runner import create_task_runner_with_rollout_fn
from rollout_async import mle_bench_rollout_fn

ConfiguredRunner = create_task_runner_with_rollout_fn(
    rollout_fn=mle_bench_rollout_fn,
    val_rollout_fn=mle_bench_rollout_fn,
)

runner = ConfiguredRunner.remote()
ray.get(runner.run.remote(config))
```

---

## Key Files in rllm

| File | Purpose |
|------|---------|
| `rllm/experimental/fully_async/runner.py` | `FullyAsyncTaskRunner`, `create_task_runner_with_rollout_fn()` |
| `rllm/experimental/fully_async/rollout_executor.py` | `RolloutExecutor` Ray actor |
| `rllm/experimental/fully_async/client.py` | `RolloutClient` async HTTP client |
| `rllm/experimental/fully_async/message_queue.py` | `MessageQueue` Ray actor |
| `rllm/experimental/fully_async/fully_async_trainer.py` | `FullyAsyncTrainer` Ray actor |
| `rllm/experimental/fully_async/param_sync.py` | `ParameterSynchronizer` |
| `rllm/experimental/fully_async/inference_manager.py` | `InferenceManager` (SGLang) |
