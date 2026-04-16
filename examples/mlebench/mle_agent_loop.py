"""Unified MLE-bench agent loop for eval and training.

This module provides a single async agent loop (MLEBenchAgent) used by both
eval.py and train.py. The only difference is the LLM client passed in:
- Eval: EvalClient (wraps OpenAI/Azure API)
- Training: RolloutClient (from rllm fully_async, has logprobs/token IDs)

Both clients satisfy the LLMClient protocol via structural subtyping.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Add mle_agent to path (same as eval.py)
sys.path.insert(0, "/home/winnieyangwn/rllm/agenthub/mle_agent")

from mle_agent.prompts import DATA_INFO_COMMAND, FORMAT_ERROR_MSG
from mle_agent.tools import execute_tool, get_tools

from rllm.experimental.fully_async.protocol import Trajectory as TrainingTrajectory
from rllm.types import Step

logger = logging.getLogger(__name__)


# =============================================================================
# Error types
# =============================================================================


class LLMCallError(Exception):
    """Raised by any LLM client when a call fails."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class LLMOutput(Protocol):
    def to_sequence(self) -> Any: ...
    def get_completion_tokens(self) -> int: ...
    def get_input_context_size(self) -> int: ...


@runtime_checkable
class LLMClient(Protocol):
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], LLMOutput]: ...


# =============================================================================
# AgentResult
# =============================================================================


@dataclass
class AgentResult:
    """Unified return type from MLEBenchAgent.run()."""

    messages: list[dict]
    steps: list[Step]
    pred_solution: str | None
    metrics: dict = field(default_factory=dict)
    trajectory: TrainingTrajectory | None = None


# =============================================================================
# EvalClient / EvalOutput
# =============================================================================


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
        """Input context size (prompt tokens only) for overflow detection."""
        return self.usage.prompt_tokens if self.usage else 0


class EvalClient:
    """Adapts OpenAI API to match RolloutClient.chat_completion() interface."""

    def __init__(self, openai_client, model: str, reasoning_effort: str | None = None, timeout: float = 600.0):
        self.client = openai_client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], EvalOutput]:
        import openai

        params = sampling_params or {}
        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": params.get("temperature", 1.0),
        }
        if params.get("top_p") is not None:
            api_kwargs["top_p"] = params["top_p"]
        if self.reasoning_effort:
            api_kwargs["reasoning_effort"] = self.reasoning_effort

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self.client.chat.completions.create, **api_kwargs),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as e:
            raise LLMCallError(f"LLM call timed out after {self.timeout}s", retryable=True) from e
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


# =============================================================================
# Prompt construction
# =============================================================================


def build_initial_messages(
    task: dict,
    sandbox: Any,
    submit_file: str = "csv",
    session_timeout: float = 360.0,
    context_size: int = 131072,
    rollout_timeout: float = 32400.0,
    max_turns: int = 128,
) -> list[dict]:
    """Build initial conversation messages for MLE-bench task.

    This is a synchronous function — it calls sandbox.exec() which blocks.
    Callers in async contexts should wrap with asyncio.to_thread().

    Args:
        task: Task dict with 'instance_id', 'task_description', etc.
        sandbox: Initialized sandbox with competition data mounted.
        submit_file: "csv" or "code" mode.
        session_timeout: Timeout for bash commands (seconds).
        context_size: Max context window size.
        rollout_timeout: Total wall-clock budget (seconds).
        max_turns: Maximum number of LLM turns.

    Returns:
        Initial messages list: [system_prompt, user_prompt]
    """
    # Get data info from sandbox
    _data_info = sandbox.exec(DATA_INFO_COMMAND, timeout=session_timeout)  # noqa: F841

    # Select prompts based on submit_file mode
    if submit_file == "code":
        from mle_agent.prompts_code import INSTANCE_PROMPT, SYSTEM_PROMPT

        system_prompt = SYSTEM_PROMPT.format(
            timeout_min=int(session_timeout / 60),
            context_size=context_size,
            eval_timeout_hrs=int(rollout_timeout / 3600),
            max_turns=max_turns,
        )
    else:
        from mle_agent.prompts_csv import INSTANCE_PROMPT, SYSTEM_PROMPT

        system_prompt = SYSTEM_PROMPT.format(
            timeout_min=int(session_timeout / 60),
            timeout_unit="minutes",
            context_size=context_size,
            rollout_timeout_hrs=int(rollout_timeout / 3600),
            max_turns=max_turns,
        )

    instance_prompt = INSTANCE_PROMPT.format(
        task_description=task.get("task_description", ""),
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance_prompt},
    ]


# =============================================================================
# MLEBenchAgent
# =============================================================================


class MLEBenchAgent:
    """Async agent loop for MLE-bench tasks.

    Used by both eval and training — the only difference is the LLM client.
    """

    def __init__(
        self,
        client: LLMClient,
        sandbox: Any,
        max_turns: int = 128,
        session_timeout: float = 360.0,
        rollout_timeout: float = 32400.0,
        context_size: int = 131072,
        context_safety_margin: float = 0.95,
        max_format_retries: int = 3,
        max_retries: int = 3,
        retry_base_delay: float = 5.0,
        retry_max_delay: float = 60.0,
        sampling_params: dict | None = None,
        check_submission_validity: bool = True,
        task_id: str = "",
        mle_bench_data_dir: str = "/checkpoint/maui/shared/cache/dojo/tasks/mlebench",
        submit_file: str = "csv",
    ):
        self.client = client
        self.sandbox = sandbox
        self.max_turns = max_turns
        self.session_timeout = session_timeout
        self.rollout_timeout = rollout_timeout
        self.context_size = context_size
        self.context_safety_margin = context_safety_margin
        self.max_format_retries = max_format_retries
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.sampling_params = sampling_params or {}
        self.task_id = task_id
        self.mle_bench_data_dir = mle_bench_data_dir
        self.submit_file = submit_file
        self.tools = get_tools(submit_file=submit_file, check_submission_validity=check_submission_validity)

    async def _call_llm_with_retry(self, messages, sampling_params, tools):
        """Call LLM with exponential backoff retry for transient errors.

        Catches both mle_agent_loop.LLMCallError (from EvalClient) and
        rllm.experimental.fully_async.client.LLMCallError (from RolloutClient).
        """
        # Import RolloutClient's LLMCallError to catch both exception types
        try:
            from rllm.experimental.fully_async.client import LLMCallError as RolloutLLMCallError
        except ImportError:
            RolloutLLMCallError = LLMCallError  # Fallback to local if import fails

        last_error = None
        for attempt in range(1 + self.max_retries):
            try:
                return await self.client.chat_completion(
                    messages=messages,
                    sampling_params=sampling_params,
                    tools=tools,
                )
            except (LLMCallError, RolloutLLMCallError) as e:
                last_error = e
                retryable = getattr(e, "retryable", False)
                if not retryable or attempt == self.max_retries:
                    raise
                delay = min(self.retry_base_delay * (2**attempt), self.retry_max_delay)
                logger.warning(
                    "LLM call failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1,
                    1 + self.max_retries,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
        raise last_error  # Unreachable, but satisfies type checker

    async def run(self, messages: list[dict]) -> AgentResult:
        """Run the async agent loop.

        Args:
            messages: Initial conversation messages (system + user prompt).

        Returns:
            AgentResult with steps, messages, pred_solution, metrics, and
            optionally a training trajectory.
        """
        steps: list[Step] = []
        trajectory = TrainingTrajectory(sequences=[])
        completion_tokens = 0
        context_size = 0  # Context window size on last turn
        last_completion_tokens = 0  # Completion tokens on last turn
        termination_reason = "max_turns"
        format_error_count = 0
        pred_solution = None
        start_time = time.monotonic()
        turn = -1

        for turn in range(self.max_turns):
            # Check rollout timeout
            elapsed = time.monotonic() - start_time
            if elapsed >= self.rollout_timeout:
                logger.warning("Rollout timeout reached after %.1f seconds", elapsed)
                termination_reason = "rollout_timeout"
                break

            # --- LLM call with retry + error handling ---
            # Catch both local LLMCallError and RolloutClient's LLMCallError
            try:
                from rllm.experimental.fully_async.client import LLMCallError as RolloutLLMCallError
            except ImportError:
                RolloutLLMCallError = LLMCallError

            try:
                msg, output = await self._call_llm_with_retry(
                    messages=messages,
                    sampling_params=self.sampling_params,
                    tools=self.tools,
                )
            except (LLMCallError, RolloutLLMCallError) as e:
                logger.warning("LLM call failed after retries on turn %d: %s", turn, e)
                termination_reason = "model_call_error"
                break

            # --- Token tracking (uniform interface) ---
            last_completion_tokens = output.get_completion_tokens()
            completion_tokens += last_completion_tokens
            context_size = output.get_input_context_size()

            effective_limit = self.context_size * self.context_safety_margin
            if context_size > effective_limit:
                logger.warning("Context size exceeded: %d > %d", context_size, int(effective_limit))
                termination_reason = "context_exceeded"
                break

            # --- Trajectory building (only when client provides token-level data) ---
            seq = output.to_sequence()
            if seq is not None:
                trajectory.append(seq)

            # --- Tool call parsing & execution ---
            messages.append(msg)
            if not msg.get("tool_calls"):
                # Format error recovery
                format_error_count += 1
                if format_error_count >= self.max_format_retries:
                    termination_reason = "format_error"
                    break
                messages.append({"role": "user", "content": FORMAT_ERROR_MSG})
                continue

            tool_parse_failed = False
            for tool_call in msg["tool_calls"]:
                tool_name = tool_call["function"]["name"]

                # Parse tool arguments — malformed JSON is treated as a format error
                try:
                    tool_args = json.loads(tool_call["function"]["arguments"])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Malformed tool arguments on turn %d: %s", turn, e)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": f"ERROR: Could not parse tool arguments: {e}. Please provide valid JSON arguments.",
                        }
                    )
                    format_error_count += 1
                    tool_parse_failed = True
                    if format_error_count >= self.max_format_retries:
                        termination_reason = "format_error"
                    break

                output_str, is_terminal, solution = await asyncio.to_thread(
                    execute_tool,
                    self.sandbox,
                    tool_name,
                    tool_args,
                    task_id=self.task_id,
                    session_timeout=self.session_timeout,
                    mle_bench_data_dir=self.mle_bench_data_dir,
                    submit_file=self.submit_file,
                )

                steps.append(Step(input={"tool": tool_name, **tool_args}, output=output_str))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": output_str,
                    }
                )

                if is_terminal:
                    pred_solution = solution
                    termination_reason = "submit"
                    break

            # Reset format error count only when all tool calls parsed successfully
            if not tool_parse_failed:
                format_error_count = 0

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
                "completion_tokens": completion_tokens,  # Sum across all turns
                "context_size": context_size,  # Context window on last turn
                "total_tokens": context_size + last_completion_tokens,  # Total conversation length
                "num_turns": max(0, turn + 1),
                "termination_reason": termination_reason,
                "rollout_duration": duration,
            },
            trajectory=final_trajectory,
        )
