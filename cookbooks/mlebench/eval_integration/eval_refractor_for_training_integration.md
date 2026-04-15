# Eval Refactor for Training Integration

**Goal:** Unify the rollout code so eval and fully async training use the exact same agent loop.

**Status:** Plan (v2 — revised)

---

## Problem

Currently eval and training have two separate agent loop implementations:

| | Eval (`examples/mlebench/eval.py`) | Training (plan in `cookbooks/mlebench/train_integration/fully_async.md`) |
|---|---|---|
| Agent loop | `_run_agent_loop()` in `mle_agent/agent.py` (sync) | `MLEBenchAgent.run()` in new `agent.py` (async) |
| LLM client | `openai.AzureOpenAI` | `RolloutClient` (rLLM async HTTP to SGLang) |
| Return type | `list[Step]` + messages + pred_solution | `protocol.Trajectory` (token-level: prompt_ids, response_ids, logprobs, masks) |
| Loop control | Format error recovery (3 retries), context size tracking | No format recovery, no context tracking |
| Tool execution | Sync `execute_tool()` | `asyncio.to_thread(execute_tool, ...)` |

**What's already shared:** tools (`mle_agent/tools.py`), prompts (`mle_agent/prompts_csv.py`, `prompts_code.py`), evaluator (`mle_agent/evaluator.py`), sandbox setup.

**What diverges:** The agent loop itself, and how the LLM is called.

**Known bug:** `MLEAgentFlow.run()` (line 364 of `agent.py`) unpacks only 3 return values from `_run_agent_loop()`, which returns 4. This causes a `ValueError` at runtime. Must be fixed before or during this refactor.

**Decision: Deprecate `MLEAgentFlow` entirely.** Its responsibilities will be split:
- **Sandbox lifecycle:** Managed by callers (`eval.py`, `train.py`) directly
- **Prompt construction:** New `build_initial_messages()` helper function
- **Agent loop:** New `MLEBenchAgent` class

---

## Solution: One Agent, Two Clients

Write one `MLEBenchAgent` class (async) used by both eval and training. The only difference is the LLM client passed in. **Eval keeps its existing `ThreadPoolExecutor` orchestration** — we only unify the inner agent loop, not the outer concurrency model.

```
                     +----------------------+
                     |  MLEBenchAgent.run()  |  <-- ONE implementation
                     |  (async agent loop)   |
                     +----------+-----------+
                                |
                    +-----------+-----------+
                    |                       |
            +-------+-------+       +-------+-------+
            | RolloutClient |       |  EvalClient   |
            | (SGLang, has  |       | (wraps OpenAI |
            |  logprobs,    |       |  Azure, same  |
            |  versions)    |       |  interface)   |
            +---------------+       +---------------+
                 Training               Eval
```

### Client Protocol

Both clients MUST satisfy this interface:

```python
class LLMClient(Protocol):
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], LLMOutput]
```

**No `cur_version` in the protocol.** Version tracking is a training-only concern — the agent loop never reads it. Training code that needs `cur_version` accesses it through the concrete `RolloutClient` type.

Where `LLMOutput` satisfies:

```python
class LLMOutput(Protocol):
    def to_sequence(self) -> Sequence | None: ...
    def get_completion_tokens(self) -> int: ...
    def get_input_context_size(self) -> int: ...
```

**Note on `get_input_context_size()`:** Returns the number of tokens in the input context (the conversation history sent to the model), NOT including the completion tokens. This is the correct metric for context overflow detection — we break when the input exceeds the model's context window, not when input+output exceeds it.

### Return Type Contract

`MLEBenchAgent.run()` returns a unified result that both eval and training can consume:

```python
from rllm.types import Step
from rllm.experimental.fully_async.protocol import Trajectory as TrainingTrajectory

@dataclass
class AgentResult:
    """Unified return type from MLEBenchAgent.run()."""
    messages: list[dict]                    # Full conversation history
    steps: list[Step]                       # rllm Step objects (tool name, args, output) — always populated
    pred_solution: str | None               # Content of train.py if submitted
    metrics: dict                           # {completion_tokens, last_input_context_size, num_turns, termination_reason, rollout_duration}
    trajectory: TrainingTrajectory | None   # Token-level trajectory — only populated when to_sequence() returns non-None
```

**Important:** `rllm.types.Trajectory` (for eval) and `rllm.experimental.fully_async.protocol.Trajectory` (for training) are **different types**. We use explicit import alias `TrainingTrajectory` to avoid confusion. The `trajectory` field here is the training-specific type with `prompt_ids`, `response_ids`, `logprobs`, and `masks`.

- **Eval** reads `steps`, `messages`, `pred_solution`, `metrics` (same data as before).
- **Training** reads `trajectory` and `metrics`.
- `steps` are always built from the message history — zero extra cost and useful for logging in both modes.

### Error Handling: Common Exception Type + Retry with Exponential Backoff

Each client normalizes its errors into `LLMCallError`, so the agent loop has a single catch target with no leaked dependencies:

```python
class LLMCallError(Exception):
    """Raised by any LLM client when a call fails (timeout, network, rate limit, etc.)."""
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable
```

The `retryable` flag lets clients signal whether a failure is worth retrying:

| Failure | Retryable? | Rationale |
|---|---|---|
| 429 rate limit | Yes | Transient, will succeed after backoff |
| 503 service unavailable | Yes | Transient server-side issue |
| 500 internal server error | Yes | Often transient |
| Timeout | Yes | May succeed on retry (especially for reasoning models) |
| Network error (connection reset, DNS) | Yes | Usually transient |
| 400 bad request | **No** | Client-side bug, retrying won't help |
| 401/403 auth error | **No** | Credentials are wrong, retrying won't help |
| Weight sync abort | N/A | Handled internally by `RolloutClient` (auto-retry, returns `finish_reason="abort"`) |

#### Retry in the agent loop

Retry logic lives in the agent loop, not in the clients. This keeps clients simple (raise and forget) and gives the agent loop visibility into retry behavior for logging and metrics.

```python
# MLEBenchAgent constructor params:
#   max_retries: int = 3          # Max retries per LLM call
#   retry_base_delay: float = 5.0 # Base delay in seconds (doubles each retry)
#   retry_max_delay: float = 60.0 # Cap on backoff delay

async def _call_llm_with_retry(self, messages, sampling_params, tools):
    """Call LLM with exponential backoff retry for transient errors."""
    last_error = None
    for attempt in range(1 + self.max_retries):
        try:
            return await self.client.chat_completion(
                messages=messages,
                sampling_params=sampling_params,
                tools=tools,
            )
        except LLMCallError as e:
            last_error = e
            if not e.retryable or attempt == self.max_retries:
                raise
            delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{1 + self.max_retries}), "
                f"retrying in {delay:.0f}s: {e}"
            )
            await asyncio.sleep(delay)
    raise last_error  # Unreachable, but satisfies type checker
```

The agent loop then calls `_call_llm_with_retry` instead of `self.client.chat_completion` directly:

```python
try:
    msg, output = await self._call_llm_with_retry(
        messages=messages,
        sampling_params=self.sampling_params,
        tools=self.tools,
    )
except LLMCallError as e:
    logger.warning(f"LLM call failed after {self.max_retries} retries on turn {turn}: {e}")
    termination_reason = "model_call_error"
    break
```

**Backoff schedule (defaults):** 5s → 10s → 20s (3 retries, base 5s). For a 429 rate limit, this gives the API ~35 seconds total to recover. The cap at 60s prevents absurd delays if `max_retries` is set high.

**Why retry in the loop, not in the client:**
1. **Visibility.** The agent loop can log retry attempts with turn context and track total retries in metrics.
2. **Uniform behavior.** Both `EvalClient` and `RolloutClient` get the same retry policy without duplicating the logic.
3. **Testability.** Unit tests can set `max_retries=0` to disable retries, or script `DummyClient` to fail N times then succeed.

**Note on EvalClient + OpenAI SDK built-in retries:** The OpenAI SDK has its own retry logic (default 2 retries for 429/500/503). Since `EvalClient` catches *all* exceptions and re-raises as `LLMCallError`, the SDK's internal retries fire first, then our loop retries on top. To avoid double-retrying, set `max_retries=0` on the OpenAI client constructor:

```python
client = openai.AzureOpenAI(..., max_retries=0)  # We handle retries ourselves
```

This gives us one consistent retry policy instead of two stacked ones with different backoff curves.

No `httpx` import in the agent loop file. Each client owns its own exception handling.

**Note:** This requires a thin wrapper around `RolloutClient.chat_completion()` to catch httpx exceptions and re-raise as `LLMCallError`. This can be done in `train.py` or as a small adapter (~10 lines), not a change to the rllm library.

### EvalClient Adapter (~50 lines)

Wraps `openai.AzureOpenAI` to match `RolloutClient.chat_completion()` signature:

```python
class EvalClient:
    """Adapts OpenAI API to match RolloutClient.chat_completion() interface."""

    def __init__(self, openai_client, model, reasoning_effort=None, timeout=600.0):
        self.client = openai_client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout  # Timeout per LLM call (seconds)

    async def chat_completion(self, messages, sampling_params=None, tools=None):
        params = sampling_params or {}
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": params.get("temperature", 1.0),
            "top_p": params.get("top_p", 1.0),
        }
        if self.reasoning_effort:
            api_kwargs["reasoning_effort"] = self.reasoning_effort

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self.client.chat.completions.create, **api_kwargs),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise LLMCallError(f"LLM call timed out after {self.timeout}s", retryable=True)
        except openai.RateLimitError as e:
            raise LLMCallError(f"Rate limited: {e}", retryable=True) from e
        except openai.APIStatusError as e:
            retryable = e.status_code in (500, 502, 503, 504)
            raise LLMCallError(f"API error {e.status_code}: {e}", retryable=retryable) from e
        except openai.APIConnectionError as e:
            raise LLMCallError(f"Connection error: {e}", retryable=True) from e
        except Exception as e:
            raise LLMCallError(f"LLM call failed: {e}", retryable=False) from e

        msg = response.choices[0].message.model_dump(exclude_none=True)
        usage = response.usage
        return msg, EvalOutput(usage=usage)
```

**Note on timeout default:** Set to 600s (10 min), not 300s. Reasoning models (o1, o3) with high `reasoning_effort` routinely take 5-10 minutes per call. This is configurable per-model via the constructor.

### EvalOutput (~30 lines)

Stand-in for `OutputWithVersion` during eval. Does NOT fake token IDs — instead exposes token counts directly.

```python
class EvalOutput:
    """Stand-in for OutputWithVersion when not training (no logprobs/token IDs).

    Unlike OutputWithVersion, this never produces Sequences or fake token IDs.
    Token counting and context tracking use explicit methods instead.
    """

    def __init__(self, usage=None):
        self.usage = usage

    def to_sequence(self):
        """No training data during eval."""
        return None

    def get_completion_tokens(self) -> int:
        """Actual completion token count from the API."""
        return self.usage.completion_tokens if self.usage else 0

    def get_input_context_size(self) -> int:
        """Input context size (prompt tokens only) for overflow detection.

        This is the number of tokens in the conversation history sent to the model.
        We use prompt_tokens (not prompt + completion) because context overflow
        happens when the INPUT exceeds the window, not output.
        """
        return self.usage.prompt_tokens if self.usage else 0
```

**Why no `all_response_ids()` returning fake IDs:** `Trajectory.merge()` calls `Sequence.is_prefix()` which does element-wise comparison of token IDs. Fake IDs (e.g., `list(range(n))`) would silently produce incorrect results if any code path accidentally calls `merge()` on eval data. Keep the abstraction honest — eval doesn't have token IDs, so don't pretend it does.

### Corresponding `OutputWithVersion` Extension

Add these methods to `OutputWithVersion` (or a thin wrapper) so both clients satisfy the same protocol:

```python
# In protocol.py or in the agent module as a wrapper:
def get_completion_tokens(self) -> int:
    return len(self.all_response_ids())

def get_input_context_size(self) -> int:
    """Input context size (prompt tokens only)."""
    return len(self.prompt_ids)
```

This way the agent loop uses one consistent API:

```python
completion_tokens += output.get_completion_tokens()
last_input_context_size = output.get_input_context_size()
if last_input_context_size > self.context_size * self.context_safety_margin:
    termination_reason = "context_exceeded"
    break
```

No `hasattr` checks, no `isinstance` branching.

**Important note on metric semantics:**
- **Eval (`EvalOutput`):** `get_completion_tokens()` returns actual API-billed tokens; `get_input_context_size()` returns exact `prompt_tokens` from OpenAI usage stats
- **Training (`OutputWithVersion`):** `get_completion_tokens()` returns `len(response_ids)`; `get_input_context_size()` returns `len(prompt_ids)` from `apply_chat_template`

These are semantically equivalent but may differ slightly because OpenAI's tokenization and chat template overhead can vary. For training, we apply a 95% safety margin (configurable via `context_safety_margin`) to account for this approximation.

### MLEBenchAgent.run() Core Loop

```python
async def run(self, messages: list[dict]) -> AgentResult:
    steps = []
    trajectory = TrainingTrajectory()
    completion_tokens = 0
    last_input_context_size = 0
    termination_reason = "max_turns"
    format_error_count = 0
    pred_solution = None
    start_time = time.monotonic()

    for turn in range(self.max_turns):
        # --- LLM call with retry + error handling ---
        try:
            msg, output = await self._call_llm_with_retry(
                messages=messages,
                sampling_params=self.sampling_params,
                tools=self.tools,
            )
        except LLMCallError as e:
            logger.warning(f"LLM call failed after retries on turn {turn}: {e}")
            termination_reason = "model_call_error"
            break

        # --- Token tracking (uniform interface) ---
        completion_tokens += output.get_completion_tokens()
        last_input_context_size = output.get_input_context_size()

        effective_limit = self.context_size * self.context_safety_margin
        if last_input_context_size > effective_limit:
            logger.warning(f"Context size exceeded: {last_input_context_size} > {effective_limit}")
            termination_reason = "context_exceeded"
            break

        # --- Trajectory building (only when client provides token-level data) ---
        seq = output.to_sequence()
        if seq is not None:
            trajectory.append(seq)

        # --- Tool call parsing & execution ---
        messages.append(msg)
        if not msg.get("tool_calls"):
            # Format error recovery (ported from _run_agent_loop)
            format_error_count += 1
            if format_error_count >= self.max_format_retries:
                termination_reason = "format_error"
                break
            messages.append({"role": "user", "content": FORMAT_ERROR_MSG})
            continue

        format_error_count = 0  # Reset on successful tool call

        for tool_call in msg["tool_calls"]:
            tool_name = tool_call["function"]["name"]

            # Parse tool arguments — malformed JSON is treated as a format error
            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Malformed tool arguments on turn {turn}: {e}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": f"ERROR: Could not parse tool arguments: {e}. Please provide valid JSON arguments.",
                })
                format_error_count += 1
                if format_error_count >= self.max_format_retries:
                    termination_reason = "format_error"
                break

            output_str, is_terminal, solution = await asyncio.to_thread(
                execute_tool, self.sandbox, tool_name, tool_args, ...
            )

            steps.append(Step(input={"tool": tool_name, **tool_args}, output=output_str))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": output_str
            })

            if is_terminal:
                pred_solution = solution
                termination_reason = "submit"
                break

        if termination_reason in ("submit", "format_error"):
            break

    duration = time.monotonic() - start_time

    # trajectory is only meaningful if it has sequences (i.e., training mode)
    final_trajectory = trajectory if trajectory.sequences else None

    return AgentResult(
        messages=messages,
        steps=steps,
        pred_solution=pred_solution,
        metrics={
            "completion_tokens": completion_tokens,
            "last_input_context_size": last_input_context_size,
            "num_turns": turn + 1,
            "termination_reason": termination_reason,
            "rollout_duration": duration,
        },
        trajectory=final_trajectory,
    )
```

**Key differences from the v1 plan:**

1. **`last_input_context_size` replaces cumulative `prompt_tokens`.** We only track the most recent turn's input context size — this is the only value needed for context overflow detection. Accumulating per-turn input sizes produced a misleading metric (each turn's input includes the full conversation history, so summing them double/triple/N-counts earlier messages). `completion_tokens` is still accumulated since each turn's completion is incremental.

2. **`rollout_duration` is tracked** via `time.monotonic()`, matching the existing `_run_agent_loop()` behavior.

3. **No `training_mode` flag.** Instead, we always create a `TrainingTrajectory` and check if it has any sequences at the end. If `to_sequence()` always returns `None` (eval mode), the trajectory stays empty and we return `None`. This avoids a boolean mode flag while achieving the same result.

4. **No `_normalize_tool_calls()`.** Both `EvalClient` (via OpenAI SDK) and `RolloutClient` (via `parse_response()`) already produce well-formed tool call dicts with `id` and `type` fields. If a client returns malformed data, that's a client bug — let it fail loudly rather than silently patching it.

5. **Single exception type** (`LLMCallError`) instead of catching 4 different exception classes.

6. **Malformed tool arguments are handled gracefully.** `json.loads()` on tool call arguments is wrapped in try/except. On failure, an error message is sent back as the tool response (so the model can self-correct), and `format_error_count` is incremented. If the model keeps producing malformed JSON and exhausts `max_format_retries`, the rollout terminates with `termination_reason="format_error"`. This prevents a single bad JSON blob from crashing the entire rollout.

---

## Design Decisions (Resolved)

### Format Error Recovery: YES, include it

Keeps eval behavior unchanged, and during training the model can learn from recovered trajectories rather than short-circuiting. The retry count is configurable via `max_format_retries` (default 3).

### Context Size Tracking: YES, via unified `get_input_context_size()` method

Both clients report **input** context size (prompt tokens only) through the same method. No `hasattr` checks or `isinstance` branching:
- `EvalClient` → exact count from `usage.prompt_tokens`
- `RolloutClient` → `len(prompt_ids)` from `apply_chat_template`

**Why input-only (not input+output):** Context overflow happens when the conversation history (input) exceeds the model's window. The completion tokens are new output being generated, not part of the overflow concern.

**Safety margin:** The two measurements may differ slightly (OpenAI API vs. `apply_chat_template` tokenization). Apply a 95% safety margin:
```python
effective_limit = self.context_size * 0.95  # Default, configurable
```

### Relationship to `MLEAgentFlow`: DEPRECATE ENTIRELY

**Decision: Deprecate `MLEAgentFlow`.** Keeping two abstractions (`MLEAgentFlow` + `MLEBenchAgent`) is confusing and adds maintenance burden.

`MLEAgentFlow`'s responsibilities will be split:

| Responsibility | New Location |
|---|---|
| Sandbox creation/destruction | `eval.py` / `train.py` directly |
| Prompt construction (SYSTEM_PROMPT, INSTANCE_PROMPT, data info) | `build_initial_messages()` helper in `mle_agent_loop.py` |
| Agent loop | `MLEBenchAgent.run()` |
| Eval orchestration | `eval.py` |

**Migration path:**
1. Phase 0: Fix `MLEAgentFlow` unpacking bug (line 364)
2. Phase 1: Build `MLEBenchAgent` alongside existing code
3. Phase 2: Update `eval.py` to use `MLEBenchAgent` directly
4. Phase 3: After validation, delete `MLEAgentFlow` from `agenthub/mle_agent/mle_agent/agent.py`

### Prompt Construction: New `build_initial_messages()` Helper

Currently `MLEAgentFlow.run()` constructs the initial messages. Extract this to a standalone function:

```python
def build_initial_messages(
    task: dict,
    sandbox: Sandbox,
    submit_file: str = "csv",
    session_timeout: float = 360.0,
) -> list[dict]:
    """Build initial conversation messages for MLE-bench task.

    This is a synchronous function — it calls sandbox.exec() which blocks.
    Callers in async contexts should wrap with asyncio.to_thread():

        messages = await asyncio.to_thread(
            build_initial_messages, task, sandbox, submit_file, session_timeout
        )

    Args:
        task: Task dict with 'instance_id', 'task_description', etc.
        sandbox: Initialized sandbox with competition data mounted
        submit_file: "csv" or "code" mode
        session_timeout: Timeout for data info command

    Returns:
        Initial messages list: [system_prompt, user_prompt]
    """
    # Get data info from sandbox (same as current MLEAgentFlow)
    data_info_output = sandbox.exec(DATA_INFO_COMMAND, timeout=session_timeout)

    # Select prompts based on submit_file mode
    if submit_file == "code":
        from mle_agent.prompts_code import SYSTEM_PROMPT, INSTANCE_PROMPT
    else:
        from mle_agent.prompts_csv import SYSTEM_PROMPT, INSTANCE_PROMPT

    user_content = INSTANCE_PROMPT.format(
        task_description=task["task_description"],
        data_info=data_info_output,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
```

This is called by `eval.py` and `train.py` before invoking `MLEBenchAgent.run(messages)`. The function is intentionally synchronous — `sandbox.exec()` is a blocking call, and making the function async would just add `await asyncio.to_thread()` internally, which is better left to the caller so they can choose how to handle it.

### Tool Call Format: Both clients already produce compatible output

Reviewed `rllm/experimental/fully_async/message_utils.py::parse_response()`:

**OpenAI API format:**
```python
{
    "role": "assistant",
    "content": "...",
    "tool_calls": [{"id": "call_abc123xyz", "type": "function", "function": {"name": "...", "arguments": "..."}}]
}
```

**RolloutClient format (from `parse_response`):**
```python
{
    "role": "assistant",
    "content": "...",
    "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "...", "arguments": "..."}}]
}
```

**Differences:**
- `id`: OpenAI uses UUID-like strings; RolloutClient uses synthetic sequential IDs (`call_0`, `call_1`, ...)
- Both include `type: "function"` field

**Resolution: No normalization needed.** The tool response message just needs to match the `tool_call_id` — doesn't matter if it's a UUID or `call_0`. Both clients produce valid `id` and `type` fields. If a future client breaks this contract, the resulting `KeyError` is a clear signal to fix the client, not to add defensive patches in the agent loop.

---

## Eval Orchestration: Keep ThreadPoolExecutor

**Decision: Do NOT convert `eval.py` to async orchestration.** The goal of this refactor is to unify the agent loop, not to rewrite the eval infrastructure.

### Why keep sync orchestration for eval

1. **Lower risk.** `eval.py`'s `ThreadPoolExecutor` works. Rewriting it to `asyncio.gather()` + `Semaphore` is a large blast-radius change that's orthogonal to the agent loop unification.
2. **The agent loop is async, but callers can bridge easily.** Each thread calls `asyncio.run(agent.run(messages))` — this is a one-line bridge from sync to async.
3. **Sandbox creation is already sync.** The `ThreadPoolExecutor` naturally limits concurrency via `max_workers`, which maps directly to max concurrent sandboxes.

### How eval.py calls the async agent loop

```python
def run_single_rollout(task_data, sample_idx, cfg):
    """Called in a ThreadPoolExecutor thread — same as today."""
    sandbox = create_sandbox(...)
    try:
        client = EvalClient(openai_client, model=cfg.model, ...)
        messages = build_initial_messages(task, sandbox, ...)

        # Bridge sync → async for the agent loop only
        result = asyncio.run(
            MLEBenchAgent(
                client=client, sandbox=sandbox,
                max_retries=3, retry_base_delay=5.0,  # Retry transient errors
                ...
            ).run(messages)
        )

        # Unpack into existing eval structures (no changes needed)
        return EvalResult(
            steps=result.steps,
            messages=result.messages,
            pred_solution=result.pred_solution,
            metrics=result.metrics,
        )
    finally:
        sandbox.close()
```

The writer thread, `ThreadPoolExecutor`, `EvalResult`, `build_trajectory_dict`, and `save_trajectory` all remain unchanged.

### Training orchestration is different (and that's fine)

Training uses `RolloutExecutor` with native async concurrency, `asyncio.Semaphore` for sandbox limiting, and `asyncio.to_thread()` for blocking calls. This is defined in `train.py`, not in the shared agent loop. The two orchestration models don't need to match — they share the agent loop, not the concurrency infrastructure.

---

## File Structure After Refactor

```
examples/mlebench/
  mle_agent_loop.py   # NEW: MLEBenchAgent, AgentResult, LLMClient/LLMOutput protocols,
                      #      EvalClient, EvalOutput, LLMCallError,
                      #      build_initial_messages()
  eval.py             # MODIFIED: uses MLEBenchAgent + EvalClient (keeps ThreadPoolExecutor)
  train.py            # NEW (from training plan): uses MLEBenchAgent + RolloutClient
  eval_ray.py         # MODIFIED: same changes as eval.py
  launch.py           # UNCHANGED
  configs/            # UNCHANGED
```

Named `mle_agent_loop.py` (not `agent.py`) to avoid confusion with `eval.py` and `train.py` in the same directory, and to clearly indicate this is the agent loop implementation.

**What lives in `mle_agent_loop.py`:**
- `LLMClient` and `LLMOutput` protocols (importable `typing.Protocol` classes)
- `LLMCallError` exception
- `MLEBenchAgent` class (the async agent loop)
- `AgentResult` dataclass
- `EvalClient` and `EvalOutput` (eval-side adapter)
- `build_initial_messages()` helper

**What does NOT live in `mle_agent_loop.py`:**
- Concurrency infrastructure (semaphores, thread pools, writers) — these belong in the caller (`eval.py` or `train.py`)
- `RolloutClient` adapter wrapping — this belongs in `train.py`

**`LLMClient` protocol lives in `mle_agent_loop.py`:** This gives a concrete, importable contract that both `EvalClient` and `RolloutClient` must satisfy. If `RolloutClient` ever drifts from the interface (e.g., a signature change in `chat_completion`), static type checkers (mypy/pyright) catch it at lint time rather than at runtime. `EvalClient` implements the protocol in the same file; `RolloutClient` doesn't need to inherit it — structural subtyping handles it — but the protocol class serves as living documentation of the contract.

`agenthub/mle_agent/` changes after Phase 3:
- **Keep:** `tools.py`, `prompts_csv.py`, `prompts_code.py`, `evaluator.py` (shared components)
- **Delete:** `_run_agent_loop()` function and `MLEAgentFlow` class from `agent.py`

---

## Implementation Steps

### Phase 0: Fix the existing bug

1. **Fix `MLEAgentFlow.run()` unpacking bug**
   - Line 364 of `agenthub/mle_agent/mle_agent/agent.py` unpacks 3 values but `_run_agent_loop()` returns 4
   - Fix: `steps, messages, pred_solution, rollout_metrics = _run_agent_loop(...)`
   - This is a standalone fix — commit it separately before starting the refactor

### Phase 1: Build the shared agent (no behavior change to eval yet)

2. **Create `examples/mlebench/mle_agent_loop.py`**
   - Define `LLMCallError` exception class
   - Define `LLMClient` and `LLMOutput` as `typing.Protocol` classes (no `cur_version` — that's training-only)
   - Implement `AgentResult` dataclass (with `rollout_duration` in metrics)
   - Implement `MLEBenchAgent` with the async agent loop (ported from `_run_agent_loop`)
     - Track `prompt_tokens` (accumulate `get_input_context_size()` per turn)
     - Track `rollout_duration` via `time.monotonic()`
     - No `training_mode` flag — infer from `to_sequence()` returning non-None
     - No `_normalize_tool_calls()` — trust clients to produce valid output
     - Catch `LLMCallError` only — clients own their exception normalization
   - Implement `EvalClient` (raises `LLMCallError` on any failure) and `EvalOutput` classes
   - Implement `build_initial_messages()` helper (sync — callers wrap in `asyncio.to_thread` if needed)

3. **Add `get_completion_tokens()` and `get_input_context_size()` to `OutputWithVersion`**
   - In `rllm/experimental/fully_async/protocol.py`, add these two methods
   - This makes `OutputWithVersion` satisfy the `LLMOutput` protocol

4. **Write unit tests with stubbed clients**

   All tests use `DummyClient` and `DummyOutput` to fully control the agent loop without any LLM or sandbox calls:

   ```python
   class DummyOutput:
       """Stubbed LLMOutput for unit testing."""
       def __init__(self, completion_tokens=100, input_context_size=1000, sequence=None):
           self._completion_tokens = completion_tokens
           self._input_context_size = input_context_size
           self._sequence = sequence
       def to_sequence(self): return self._sequence
       def get_completion_tokens(self): return self._completion_tokens
       def get_input_context_size(self): return self._input_context_size

   class DummyClient:
       """Stubbed LLM client that replays a scripted sequence of responses."""
       def __init__(self, responses: list[tuple[dict, DummyOutput]]):
           self.responses = responses
           self.call_idx = 0
           self.recorded_messages = []  # For asserting what was sent

       async def chat_completion(self, messages, sampling_params=None, tools=None):
           self.recorded_messages.append(messages.copy())
           if self.call_idx >= len(self.responses):
               raise LLMCallError("No more scripted responses")
           msg, output = self.responses[self.call_idx]
           self.call_idx += 1
           return msg, output
   ```

   `execute_tool` is also stubbed — the tests monkey-patch or inject a fake that returns `(output_str, is_terminal, solution)` without touching a real sandbox.

   #### Test cases:

   **A. Happy path — multi-turn with tool calls → submit**
   - Script: 3 turns of tool calls, then a submit tool call
   - Assert: `termination_reason == "submit"`, `num_turns == 4`, `pred_solution` is populated, `steps` has one entry per tool call, `messages` contains the full conversation

   **B. Format error recovery — no tool calls → retry → success**
   - Script: turn 1 returns a message with no `tool_calls`, turn 2 returns valid tool calls
   - Assert: `FORMAT_ERROR_MSG` was appended after turn 1, `format_error_count` resets on turn 2, final `termination_reason` is not `"format_error"`

   **C. Format error exhaustion → termination**
   - Script: N consecutive messages with no `tool_calls` (where N = `max_format_retries`)
   - Assert: `termination_reason == "format_error"`, `num_turns == N`

   **D. Malformed tool arguments → error feedback → recovery**
   - Script: turn 1 returns a tool_call with `arguments: "not valid json"`, turn 2 returns valid tool call
   - Assert: tool response message contains `"ERROR: Could not parse tool arguments"`, model gets a chance to retry, `format_error_count` incremented

   **E. Malformed tool arguments exhaustion → termination**
   - Script: repeated malformed JSON tool calls exceeding `max_format_retries`
   - Assert: `termination_reason == "format_error"`

   **F. Context size exceeded → termination**
   - Script: `DummyOutput(input_context_size=200_000)` with `context_size=128_000` and `context_safety_margin=0.95`
   - Assert: `termination_reason == "context_exceeded"`, loop breaks on that turn, no tool execution attempted

   **G. LLM transient error → retry → success**
   - Script: `DummyClient` raises `LLMCallError("rate limited", retryable=True)` on first call, succeeds on second call. Set `max_retries=3`, `retry_base_delay=0.01` (fast tests).
   - Assert: rollout continues past the error, `termination_reason != "model_call_error"`, the successful response is processed normally

   **H. LLM transient error → retries exhausted → termination**
   - Script: `DummyClient` raises `LLMCallError("service unavailable", retryable=True)` on every call. Set `max_retries=2`, `retry_base_delay=0.01`.
   - Assert: `termination_reason == "model_call_error"`, total attempts == 3 (1 initial + 2 retries)

   **I. LLM non-retryable error → immediate termination (no retry)**
   - Script: `DummyClient` raises `LLMCallError("bad request", retryable=False)` on first call. Set `max_retries=3`.
   - Assert: `termination_reason == "model_call_error"`, only 1 attempt made (no retries)

   **J. Max turns exhaustion**
   - Script: all turns return valid tool calls but never submit, `max_turns=3`
   - Assert: `termination_reason == "max_turns"`, `num_turns == 3`

   **K. Token tracking — `last_input_context_size` is last turn only**
   - Script: 3 turns with `input_context_size` of 1000, 2000, 3000 respectively
   - Assert: `metrics["last_input_context_size"] == 3000` (not 6000)
   - Assert: `metrics["completion_tokens"]` is the sum of all turns' completion tokens

   **L. `rollout_duration` is positive**
   - Any multi-turn script
   - Assert: `metrics["rollout_duration"] > 0`

   **M. `trajectory` is `None` in eval mode**
   - Script: all `DummyOutput.to_sequence()` return `None`
   - Assert: `result.trajectory is None`

   **N. `trajectory` is populated in training mode**
   - Script: `DummyOutput(sequence=mock_sequence)` where `mock_sequence` is a stub `Sequence` object
   - Assert: `result.trajectory is not None`, `len(result.trajectory.sequences) == num_turns`

   **O. Messages history is correct**
   - Script: 2 tool call turns, then submit
   - Assert: messages alternate correctly: `[system, user, assistant, tool, assistant, tool, assistant, tool]`
   - Assert: each tool response `tool_call_id` matches the corresponding assistant message's tool call `id`

5. **Write a compatibility test (replay-based)**
   - Record a transcript: capture the exact sequence of `(messages_sent, response_received)` from one run of the old `_run_agent_loop()`
   - Replay it through `MLEBenchAgent + EvalClient` using a replay client that returns the same responses
   - Compare: turn count, tool calls made, termination reason, final messages list
   - This is deterministic (no live LLM calls) and catches behavioral regressions before touching eval.py

### Phase 2: Wire into eval.py (minimal changes)

6. **Update `examples/mlebench/eval.py`**
   - Replace `_run_agent_loop()` call with `asyncio.run(MLEBenchAgent(...).run(messages))` inside `run_single_rollout()`
   - Unpack `AgentResult` into existing `EvalResult` / `RolloutOutput` structures
   - Use `build_initial_messages()` for prompt construction
   - **Keep `ThreadPoolExecutor`** — no change to concurrency model
   - **Keep writer thread** — no change to I/O model
   - Keep `EvalResult`, `build_trajectory_dict`, `save_trajectory` unchanged

7. **Update `examples/mlebench/eval_ray.py`** — same changes as eval.py

### Phase 3: Validate and Clean Up

8. **Side-by-side comparison test**
   - Run: `python eval.py --config configs/gpt5_test.yaml --task mlsp-2013-birds --samples 1`
   - Compare output against a saved baseline from the old eval.py
   - Verify: same prompts sent, same tools called, same eval results, same metrics
   - Verify: `rollout_duration` is populated and reasonable

9. **Keep `_run_agent_loop` as fallback initially**
   - Add a `--legacy-agent-loop` flag to eval.py that falls back to the old sync path
   - Keep for at least one full eval cycle

10. **Delete deprecated code after validation**
    - Remove `MLEAgentFlow` class from `agenthub/mle_agent/mle_agent/agent.py`
    - Remove `_run_agent_loop()` function
    - Remove `--legacy-agent-loop` flag
    - Update any imports/references

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `asyncio.run()` in thread creates new event loop per rollout | Low | This is expected behavior — each thread gets its own event loop. No shared state between loops. Verified: `asyncio.run()` in a `ThreadPoolExecutor` thread works correctly. |
| Behavioral regression from replacing `_run_agent_loop` | Medium | Unit tests with `DummyClient`; side-by-side comparison test; `--legacy-agent-loop` fallback |
| Transient LLM errors (429, 503, timeout) kill rollouts | Medium | Exponential backoff retry with `retryable` flag on `LLMCallError`; default 3 retries with 5s base delay; non-retryable errors (400, 401) fail immediately |
| Double retry (OpenAI SDK + our loop) | Low | Disable OpenAI SDK built-in retries (`max_retries=0` on client constructor); our loop provides the single retry policy |
| Context size measurement differs between eval and training | Low | Use 95% safety margin; `get_input_context_size()` uses input-only (not input+output) |
| OpenAI API hangs indefinitely | Low | `asyncio.wait_for()` with configurable timeout (default 600s) |
| Reasoning model calls exceed default timeout | Medium | Default 600s (not 300s); configurable per-model via `EvalClient(timeout=...)` |
| Malformed tool call JSON crashes rollout | Medium | `json.loads` wrapped in try/except; error sent back to model as tool response; counted as format error |
| Future code accidentally calls `merge()` on eval trajectories | None | `EvalOutput.to_sequence()` returns `None`; `trajectory` is `None` in eval mode |
| `RolloutClient` drifts from `LLMClient` protocol | Low | `typing.Protocol` + static type checker catches at lint time |
| Type confusion (`Trajectory` vs `TrainingTrajectory`) | Low | Explicit import alias; clear documentation in code |

---

## References

- Training plan: `cookbooks/mlebench/train_integration/fully_async.md`
- Current eval: `examples/mlebench/eval.py`
- Current agent loop: `agenthub/mle_agent/mle_agent/agent.py` (`_run_agent_loop`)
- Reference implementation: `examples/fully_async/deepresearch/` (SearchAgent + train.py pattern)
- RolloutClient interface: `rllm/experimental/fully_async/client.py`
- Protocol types: `rllm/experimental/fully_async/protocol.py`
- Message parsing: `rllm/experimental/fully_async/message_utils.py`
