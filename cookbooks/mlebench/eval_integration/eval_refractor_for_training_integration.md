# Eval Refactor for Training Integration

**Goal:** Unify the rollout code so eval and fully async training use the exact same agent loop, with a single `RolloutClient` that supports both eval (text mode) and training (token mode).

**Status:** Phase 1-3 complete. RolloutClient now supports both text mode (eval) and token mode (training).

---

## What's Already Implemented (Phase 1-2)

### Unified Agent Loop

`MLEBenchAgent` in `examples/mlebench/mle_agent_loop.py` is the single async agent loop used by both eval and training. The only difference is the LLM client passed in.

**Implemented components:**

| Component | Location | Status |
|---|---|---|
| `MLEBenchAgent` | `mle_agent_loop.py` | Done |
| `AgentResult` dataclass | `mle_agent_loop.py` | Done |
| `LLMClient` / `LLMOutput` protocols | `mle_agent_loop.py` | Done |
| `EvalClient` / `EvalOutput` | `mle_agent_loop.py` | Done (to be replaced by RolloutClient text mode) |
| `LLMCallError` exception | `mle_agent_loop.py` | Done |
| `build_initial_messages()` | `mle_agent_loop.py` | Done |
| `get_completion_tokens()` / `get_input_context_size()` on `OutputWithVersion` | `protocol.py` | Done |
| `eval.py` integration | `eval.py` | Done (uses MLEBenchAgent + EvalClient) |
| `--legacy-agent-loop` fallback | `eval.py` | Done |
| Unit tests with DummyClient | `tests/test_mle_agent_loop.py` | Done |

### Client Protocol (unchanged)

```python
class LLMClient(Protocol):
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], LLMOutput]

class LLMOutput(Protocol):
    def to_sequence(self) -> Sequence | None: ...
    def get_completion_tokens(self) -> int: ...
    def get_input_context_size(self) -> int: ...
```

### Return Type (unchanged)

```python
@dataclass
class AgentResult:
    messages: list[dict]
    steps: list[Step]
    pred_solution: str | None
    metrics: dict  # {completion_tokens, last_input_context_size, num_turns, termination_reason, rollout_duration}
    trajectory: TrainingTrajectory | None  # Only populated when to_sequence() returns non-None
```

### Error Handling (unchanged)

- `LLMCallError(message, retryable)` — common exception type
- Retry with exponential backoff in `MLEBenchAgent._call_llm_with_retry()`
- OpenAI SDK retries disabled (`max_retries=0`) to avoid double-retry

---

## Phase 3: Enhance RolloutClient with Text/Token Dual Mode

### Problem

Currently two separate clients exist:

| | `EvalClient` | `RolloutClient` |
|---|---|---|
| Backend | OpenAI SDK (Azure only) | httpx to SGLang `/generate` |
| Returns | `EvalOutput` (no token IDs) | `OutputWithVersion` (token IDs + logprobs) |
| Supports | Azure OpenAI only | SGLang only |

We want to run MLE eval against locally hosted open-source models (e.g., Qwen 3.5-4B via SGLang). Rather than adding yet another client, we enhance `RolloutClient` with two modes:

- **Token mode** (tokenizer provided): Uses SGLang's `/generate` endpoint. Returns `OutputWithVersion` with token IDs + logprobs. For training.
- **Text mode** (no tokenizer): Uses OpenAI-compatible `/v1/chat/completions` endpoint. Returns `OutputWithVersion` with `to_sequence() → None`. For eval. Works with SGLang, vLLM, or any OpenAI-compatible server.

Mode is auto-detected: `tokenizer is not None` → token mode, else → text mode.

### Architecture

```
                     +----------------------+
                     |  MLEBenchAgent.run()  |  <-- ONE implementation (already done)
                     +----------+-----------+
                                |
                    +-----------+-----------+
                    |                       |
            +-------+-------+       +-------+-------+
            | RolloutClient |       | RolloutClient |
            | (token mode)  |       | (text mode)   |
            | tokenizer=tok |       | tokenizer=None|
            | /generate     |       | /v1/chat/comp |
            +---------------+       +---------------+
                Training          Eval (SGLang/vLLM)
```

For Azure OpenAI eval, `EvalClient` remains as a backward-compatible option.

### Implementation

#### 1. Add usage overrides to `OutputWithVersion`

**File:** `rllm/experimental/fully_async/protocol.py`

In text mode we don't have token IDs — we get usage stats from the API instead. Add optional override fields so the existing getters work for both modes:

```python
@dataclass
class OutputWithVersion:
    prompt_ids: list[int]
    output_chunks: list[OutputChunk]
    finish_reason: str = "Not Finish"
    prompt_text: str = ""
    # Optional overrides for text mode (no token IDs available)
    _completion_token_count: int | None = None
    _input_context_size: int | None = None

    def to_sequence(self):
        if not self.output_chunks:  # text mode: no token data
            return None
        # ... existing logic unchanged ...

    def get_completion_tokens(self) -> int:
        if self._completion_token_count is not None:
            return self._completion_token_count
        return len(self.all_response_ids())

    def get_input_context_size(self) -> int:
        if self._input_context_size is not None:
            return self._input_context_size
        return len(self.prompt_ids)
```

Token mode callers are unaffected — the overrides default to `None` so existing behavior is preserved.

#### 2. Add text mode to `RolloutClient`

**File:** `rllm/experimental/fully_async/client.py`

```python
class RolloutClient:
    def __init__(
        self,
        router_url: str,
        tokenizer=None,          # None → text mode, provided → token mode
        max_concurrency: int = 4096,
        max_tokens=32768,
        model: str | None = None,   # NEW: required for text mode (/v1/chat/completions)
        api_key: str | None = None, # NEW: optional auth for the server
    ):
        self.router_url = router_url
        self.tokenizer = tokenizer
        self.model = model
        self.api_key = api_key
        self._mode = "token" if tokenizer else "text"

        # Only create parser in token mode (requires tokenizer)
        self.parser = ToolParser.get_parser(tokenizer) if tokenizer else None

        # ... rest unchanged (httpx client, resume_event, etc.) ...

    async def chat_completion(self, messages, sampling_params=None, tools=None):
        if self._mode == "text":
            return await self._chat_completion_text(messages, sampling_params, tools)
        else:
            return await self._chat_completion_token(messages, sampling_params, tools)

    async def _chat_completion_token(self, messages, sampling_params, tools):
        """Token mode: existing logic (renamed from chat_completion)."""
        # ... current chat_completion() body, unchanged ...

    async def _chat_completion_text(self, messages, sampling_params=None, tools=None):
        """Text mode: call /v1/chat/completions (OpenAI-compatible)."""
        params = sampling_params or {}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": params.get("temperature", 1.0),
        }
        if tools:
            payload["tools"] = tools
        if params.get("top_p") is not None:
            payload["top_p"] = params["top_p"]

        await self.resume_event.wait()

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = await self.client.post(
            self.router_url + "/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})

        output = OutputWithVersion(
            prompt_ids=[],
            output_chunks=[],
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            _completion_token_count=usage.get("completion_tokens", 0),
            _input_context_size=usage.get("prompt_tokens", 0),
        )
        return msg, output
```

**Key design decisions:**
- Auto-detect mode via `tokenizer is not None` — no explicit `mode` parameter needed
- `ToolParser` only created in token mode (it requires a tokenizer)
- Text mode uses the same httpx client — no openai SDK dependency
- `pause()` / `resume()` / `set_version()` still work in text mode (they're no-ops functionally but don't break anything)

#### 3. Update `eval.py` with backend dispatch

**File:** `examples/mlebench/eval.py`

In `run_single_rollout()`, add backend dispatch. Azure keeps existing `EvalClient` path; SGLang/vLLM uses `RolloutClient` text mode:

```python
backend = getattr(cfg.model, 'backend', 'azure')

if backend in ("sglang", "vllm"):
    from rllm.experimental.fully_async.client import RolloutClient

    client = RolloutClient(
        router_url=cfg.model.base_url,
        model=cfg.model.name,
        api_key=getattr(cfg.model, 'api_key', None),
    )
    print(f"✓ Created RolloutClient in text mode (model: {cfg.model.name}, backend: {backend})")
else:
    # Existing Azure path
    openai_client = openai.AzureOpenAI(
        azure_endpoint=cfg.model.azure_endpoint,
        api_key=cfg.model.api_key,
        api_version=cfg.model.api_version,
        max_retries=0,
    )
    client = EvalClient(openai_client=openai_client, model=cfg.model.name, reasoning_effort=reasoning_effort)
    print(f"✓ Created EvalClient (model: {cfg.model.name})")
```

Same dispatch in `_run_agent_loop_legacy()` for the legacy path.

#### 4. Add config fields

**File:** `examples/mlebench/configs/base.yaml`

Add under `model:` (with backward-compatible defaults):

```yaml
model:
  name: ""
  backend: azure        # "azure", "sglang", "vllm"
  base_url: null        # e.g. http://localhost:30000 (for sglang/vllm)
  # Azure-specific (used when backend=azure)
  azure_endpoint: ""
  api_key: ""
  api_version: ""
  reasoning_effort: null
```

#### 5. Create example SGLang config

**File:** `examples/mlebench/configs/sglang.yaml` (new)

```yaml
# MLE-bench Evaluation with SGLang-served model
#
# Start SGLang server first:
#   python -m sglang.launch_server --model-path Qwen/Qwen3-32B \
#       --tool-call-parser qwen25 --port 30000
#
# Then run eval:
#   python eval.py --config configs/sglang.yaml --task mlsp-2013-birds

defaults:
  - base

model:
  name: Qwen/Qwen3-32B
  backend: sglang
  base_url: http://localhost:30000

agent:
  context_size: 32768
  temperature: 1.0

eval:
  samples_per_prompt: 2
  output_dir: /checkpoint/maui_sft/winnieyangwn/RLLM/sglang_test
```

### Files to modify

| File | Change |
|---|---|
| `rllm/experimental/fully_async/protocol.py` | Add `_completion_token_count`, `_input_context_size` override fields; update `to_sequence()`, `get_completion_tokens()`, `get_input_context_size()` |
| `rllm/experimental/fully_async/client.py` | Add `model`, `api_key` params; add `_mode` auto-detection; add `_chat_completion_text()`; rename existing logic to `_chat_completion_token()` |
| `examples/mlebench/eval.py` | Add backend dispatch: azure → EvalClient, sglang/vllm → RolloutClient text mode |
| `examples/mlebench/configs/base.yaml` | Add `backend` and `base_url` fields |
| `examples/mlebench/configs/sglang.yaml` | New example config |

### What does NOT change

- `MLEBenchAgent` — uses `LLMClient` protocol, works with any client
- `EvalClient` / `EvalOutput` — kept for Azure backward compat
- Token mode behavior — existing `generate()` and `_generate()` unchanged
- `RolloutExecutor` — always passes tokenizer → always gets token mode
- Deepresearch example — always uses tokenizer, unaffected
- Unit tests — use `DummyClient`, unaffected

### Verification

1. **Existing token mode**: deepresearch training still works (tokenizer provided → token mode)
2. **Text mode with SGLang**:
   ```bash
   python -m sglang.launch_server --model-path Qwen/Qwen3-32B \
       --tool-call-parser qwen25 --port 30000
   python eval.py --config configs/sglang.yaml --task mlsp-2013-birds
   ```
3. **Azure backward compat**: existing `configs/gpt5.yaml` works unchanged
4. **Unit tests**: `pytest examples/mlebench/tests/test_mle_agent_loop.py`

### Risks & Mitigations

| Risk | Mitigation |
|---|---|
| SGLang tool calling format differs from OpenAI | SGLang with `--tool-call-parser qwen25` produces OpenAI-compatible format. Mismatches surface as `format_error` termination (already handled by retry logic). |
| SGLang `/v1/chat/completions` doesn't return `usage` | `usage.get("completion_tokens", 0)` defaults gracefully. Token metrics will be 0 — acceptable for eval (informational, not functional). |
| `reasoning_effort` not supported by SGLang | SGLang configs omit `reasoning_effort` (defaults to `null`). `EvalClient` guard `if self.reasoning_effort:` already handles this. RolloutClient text mode simply doesn't send it. |
| httpx errors not wrapped in `LLMCallError` | `MLEBenchAgent._call_llm_with_retry()` catches both `LLMCallError` and general `Exception`. httpx errors propagate as general exceptions and trigger `termination_reason="model_call_error"`. For better retry behavior, can add httpx error wrapping later. |
| Model name mismatch with SGLang | `cfg.model.name` must match `--model-path` exactly. Documented in example config. |

---

## Future: Eval During Training (val_rollout_fn)

Once RolloutClient dual mode is working, the next step is adding eval-during-training support following the deepresearch pattern:

1. Define `mle_val_rollout_fn(client, tokenizer, **datum)` that uses `MLEBenchAgent` + `RolloutClient` text mode
2. Pass to `AsyncAgentTrainer(config, rollout_fn, val_rollout_fn)`
3. Configure `rollout.test_freq` for validation frequency
4. Validation metrics (`val/avg_reward`, `val/percentile`, etc.) logged automatically

This is a separate effort tracked in `cookbooks/mlebench/train_integration/fully_async.md`.

---

## References

- Training plan: `cookbooks/mlebench/train_integration/fully_async.md`
- Current eval: `examples/mlebench/eval.py`
- Unified agent loop: `examples/mlebench/mle_agent_loop.py`
- Reference implementation: `examples/fully_async/deepresearch/` (SearchAgent + train.py pattern)
- RolloutClient: `rllm/experimental/fully_async/client.py`
- Protocol types: `rllm/experimental/fully_async/protocol.py`
