"""Unit tests for MLEBenchAgent using stubbed clients.

Tests cover all scenarios from the plan (A-O):
- Happy path, format errors, malformed JSON, context exceeded
- LLM errors with retry, max turns, token tracking
- Trajectory population, messages history
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

# Add mle_agent to path before importing mle_agent_loop
sys.path.insert(0, "/home/winnieyangwn/rllm/agenthub/mle_agent")
sys.path.insert(0, "/home/winnieyangwn/rllm/examples/mlebench")

from mle_agent_loop import LLMCallError, MLEBenchAgent

# =============================================================================
# Stubs
# =============================================================================


@dataclass
class DummyOutput:
    """Stubbed LLMOutput for unit testing."""

    completion_tokens: int = 100
    input_context_size: int = 1000
    sequence: object | None = None

    def to_sequence(self):
        return self.sequence

    def get_completion_tokens(self) -> int:
        return self.completion_tokens

    def get_input_context_size(self) -> int:
        return self.input_context_size


class DummyClient:
    """Stubbed LLM client that replays a scripted sequence of responses."""

    def __init__(self, responses: list[tuple[dict, DummyOutput]]):
        self.responses = responses
        self.call_idx = 0
        self.call_count = 0

    async def chat_completion(self, messages, sampling_params=None, tools=None):
        self.call_count += 1
        if self.call_idx >= len(self.responses):
            raise LLMCallError("No more scripted responses", retryable=False)
        msg, output = self.responses[self.call_idx]
        self.call_idx += 1
        return msg, output


class FailThenSucceedClient:
    """Client that fails N times then succeeds."""

    def __init__(self, fail_count: int, success_responses: list[tuple[dict, DummyOutput]], retryable: bool = True):
        self.fail_count = fail_count
        self.success_responses = success_responses
        self.call_count = 0
        self.success_idx = 0
        self.retryable = retryable

    async def chat_completion(self, messages, sampling_params=None, tools=None):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise LLMCallError(f"Transient error (attempt {self.call_count})", retryable=self.retryable)
        if self.success_idx >= len(self.success_responses):
            raise LLMCallError("No more responses", retryable=False)
        msg, output = self.success_responses[self.success_idx]
        self.success_idx += 1
        return msg, output


class DummySandbox:
    """Minimal sandbox stub for execute_tool."""

    def exec(self, command: str, timeout: float = 60.0) -> str:
        return "sandbox output"

    def fetch_file(self, remote_path: str, local_path: str) -> bool:
        return False

    def close(self):
        pass


# =============================================================================
# Helpers
# =============================================================================


def _make_tool_call_msg(tool_name: str = "bash", arguments: dict | str | None = None, call_id: str = "call_0") -> dict:
    """Create a well-formed assistant message with a tool call."""
    if arguments is None:
        arguments = {"command": "ls"}
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            }
        ],
    }


def _make_submit_msg(call_id: str = "call_submit") -> dict:
    """Create a submit tool call message."""
    return _make_tool_call_msg(
        tool_name="submit",
        arguments={"train_path": "/workspace/train.py", "submission_path": "/workspace/submission.csv"},
        call_id=call_id,
    )


def _make_no_tool_msg(content: str = "I think we should...") -> dict:
    """Create an assistant message with no tool calls."""
    return {"role": "assistant", "content": content}


def _make_agent(client, sandbox=None, **kwargs) -> MLEBenchAgent:
    """Create an MLEBenchAgent with test defaults."""
    defaults = {
        "max_turns": 10,
        "max_retries": 0,  # No retries by default in tests
        "retry_base_delay": 0.01,
        "session_timeout": 10.0,
        "rollout_timeout": 3600.0,
        "context_size": 128000,
        "context_safety_margin": 0.95,
    }
    defaults.update(kwargs)
    return MLEBenchAgent(
        client=client,
        sandbox=sandbox or DummySandbox(),
        **defaults,
    )


def _patch_execute_tool(output: str = "tool output", is_terminal: bool = False, solution: str | None = None):
    """Patch execute_tool to return fixed values."""
    return patch("mle_agent_loop.execute_tool", return_value=(output, is_terminal, solution))


# =============================================================================
# Tests
# =============================================================================


class TestHappyPath:
    """A. Normal multi-turn with tool calls → submit."""

    def test_multi_turn_submit(self):
        responses = [
            (_make_tool_call_msg(call_id="call_0"), DummyOutput()),
            (_make_tool_call_msg(call_id="call_1"), DummyOutput()),
            (_make_submit_msg(), DummyOutput()),
        ]
        client = DummyClient(responses)

        with _patch_execute_tool() as mock_et:
            # Last call is submit — make it terminal
            mock_et.side_effect = [
                ("tool output", False, None),
                ("tool output", False, None),
                ("Solution submitted.", True, "print('train')"),
            ]
            result = asyncio.run(
                _make_agent(client).run(
                    [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "task"},
                    ]
                )
            )

        assert result.metrics["termination_reason"] == "submit"
        assert result.metrics["num_turns"] == 3
        assert result.pred_solution == "print('train')"
        assert len(result.steps) == 3
        assert result.metrics["completion_tokens"] == 300  # 3 * 100


class TestFormatErrorRecovery:
    """B. Format error recovery — no tool calls → retry → success."""

    def test_recovery(self):
        responses = [
            (_make_no_tool_msg(), DummyOutput()),  # No tool calls
            (_make_tool_call_msg(call_id="call_0"), DummyOutput()),  # Valid
            (_make_submit_msg(), DummyOutput()),  # Submit
        ]
        client = DummyClient(responses)

        with _patch_execute_tool() as mock_et:
            mock_et.side_effect = [
                ("tool output", False, None),
                ("Solution submitted.", True, "train"),
            ]
            result = asyncio.run(
                _make_agent(client).run(
                    [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "task"},
                    ]
                )
            )

        assert result.metrics["termination_reason"] == "submit"
        # Check FORMAT_ERROR_MSG was sent
        user_msgs = [m for m in result.messages if m.get("role") == "user"]
        assert any("tool" in m.get("content", "").lower() for m in user_msgs)


class TestFormatErrorExhaustion:
    """C. Format error recovery exhausted → termination."""

    def test_exhaustion(self):
        responses = [
            (_make_no_tool_msg(), DummyOutput()),
            (_make_no_tool_msg(), DummyOutput()),
            (_make_no_tool_msg(), DummyOutput()),
        ]
        client = DummyClient(responses)

        result = asyncio.run(
            _make_agent(client, max_format_retries=3).run(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                ]
            )
        )

        assert result.metrics["termination_reason"] == "format_error"


class TestMalformedToolArgs:
    """D. Malformed tool arguments → error feedback → recovery."""

    def test_recovery(self):
        bad_msg = _make_tool_call_msg(arguments="not valid json", call_id="call_bad")
        good_msg = _make_submit_msg()
        responses = [
            (bad_msg, DummyOutput()),
            (good_msg, DummyOutput()),
        ]
        client = DummyClient(responses)

        with _patch_execute_tool() as mock_et:
            mock_et.return_value = ("Solution submitted.", True, "train")
            result = asyncio.run(
                _make_agent(client).run(
                    [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "task"},
                    ]
                )
            )

        assert result.metrics["termination_reason"] == "submit"
        # Check error message was sent back as tool response
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert any("ERROR: Could not parse tool arguments" in m.get("content", "") for m in tool_msgs)


class TestMalformedToolArgsExhaustion:
    """E. Malformed tool arguments exhaustion → termination."""

    def test_exhaustion(self):
        bad_msg = _make_tool_call_msg(arguments="not json", call_id="call_bad")
        responses = [
            (bad_msg, DummyOutput()),
            (bad_msg, DummyOutput()),
            (bad_msg, DummyOutput()),
        ]
        client = DummyClient(responses)

        result = asyncio.run(
            _make_agent(client, max_format_retries=3).run(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                ]
            )
        )

        assert result.metrics["termination_reason"] == "format_error"


class TestContextSizeExceeded:
    """F. Context size exceeded → termination."""

    def test_exceeded(self):
        responses = [
            (_make_tool_call_msg(), DummyOutput(input_context_size=200_000)),
        ]
        client = DummyClient(responses)

        result = asyncio.run(
            _make_agent(client, context_size=128_000, context_safety_margin=0.95).run(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                ]
            )
        )

        assert result.metrics["termination_reason"] == "context_exceeded"
        assert len(result.steps) == 0  # No tool execution attempted


class TestLLMRetrySuccess:
    """G. LLM transient error → retry → success."""

    def test_retry_then_success(self):
        submit_msg = _make_submit_msg()
        client = FailThenSucceedClient(
            fail_count=1,
            success_responses=[(submit_msg, DummyOutput())],
            retryable=True,
        )

        with _patch_execute_tool() as mock_et:
            mock_et.return_value = ("Submitted.", True, "train")
            result = asyncio.run(
                _make_agent(client, max_retries=3, retry_base_delay=0.01).run(
                    [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "task"},
                    ]
                )
            )

        assert result.metrics["termination_reason"] == "submit"
        assert client.call_count == 2  # 1 fail + 1 success


class TestLLMRetryExhaustion:
    """H. LLM transient error → retries exhausted → termination."""

    def test_exhaustion(self):
        client = FailThenSucceedClient(
            fail_count=100,  # Always fails
            success_responses=[],
            retryable=True,
        )

        result = asyncio.run(
            _make_agent(client, max_retries=2, retry_base_delay=0.01).run(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                ]
            )
        )

        assert result.metrics["termination_reason"] == "model_call_error"
        assert client.call_count == 3  # 1 initial + 2 retries


class TestLLMNonRetryableError:
    """I. LLM non-retryable error → immediate termination (no retry)."""

    def test_no_retry(self):
        client = FailThenSucceedClient(
            fail_count=100,
            success_responses=[],
            retryable=False,
        )

        result = asyncio.run(
            _make_agent(client, max_retries=3, retry_base_delay=0.01).run(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                ]
            )
        )

        assert result.metrics["termination_reason"] == "model_call_error"
        assert client.call_count == 1  # No retries


class TestMaxTurnsExhaustion:
    """J. Max turns exhaustion."""

    def test_max_turns(self):
        responses = [(_make_tool_call_msg(call_id=f"call_{i}"), DummyOutput()) for i in range(3)]
        client = DummyClient(responses)

        with _patch_execute_tool():
            result = asyncio.run(
                _make_agent(client, max_turns=3).run(
                    [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "task"},
                    ]
                )
            )

        assert result.metrics["termination_reason"] == "max_turns"
        assert result.metrics["num_turns"] == 3


class TestTokenTracking:
    """K. context_size is last turn only, not cumulative. prompt_tokens is cumulative."""

    def test_last_turn_only(self):
        responses = [
            (_make_tool_call_msg(call_id="call_0"), DummyOutput(completion_tokens=10, input_context_size=1000)),
            (_make_tool_call_msg(call_id="call_1"), DummyOutput(completion_tokens=20, input_context_size=2000)),
            (_make_tool_call_msg(call_id="call_2"), DummyOutput(completion_tokens=30, input_context_size=3000)),
        ]
        client = DummyClient(responses)

        with _patch_execute_tool():
            result = asyncio.run(
                _make_agent(client, max_turns=3).run(
                    [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "task"},
                    ]
                )
            )

        assert result.metrics["context_size"] == 3000  # Last turn only
        assert result.metrics["total_tokens"] == 3030  # context_size + last_completion = 3000 + 30
        assert result.metrics["prompt_tokens"] == 6000  # Cumulative: 1000 + 2000 + 3000
        assert result.metrics["completion_tokens"] == 60  # 10 + 20 + 30


class TestRolloutDuration:
    """L. rollout_duration is positive."""

    def test_positive_duration(self):
        responses = [(_make_submit_msg(), DummyOutput())]
        client = DummyClient(responses)

        with _patch_execute_tool() as mock_et:
            mock_et.return_value = ("Submitted.", True, "train")
            result = asyncio.run(
                _make_agent(client).run(
                    [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "task"},
                    ]
                )
            )

        assert result.metrics["rollout_duration"] > 0


class TestTrajectoryNoneInEval:
    """M. trajectory is None when to_sequence() always returns None."""

    def test_none_trajectory(self):
        responses = [(_make_submit_msg(), DummyOutput(sequence=None))]
        client = DummyClient(responses)

        with _patch_execute_tool() as mock_et:
            mock_et.return_value = ("Submitted.", True, "train")
            result = asyncio.run(
                _make_agent(client).run(
                    [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "task"},
                    ]
                )
            )

        assert result.trajectory is None


class TestTrajectoryPopulatedInTraining:
    """N. trajectory is populated when to_sequence() returns sequences."""

    def test_populated_trajectory(self):
        mock_seq = MagicMock()
        responses = [
            (_make_tool_call_msg(call_id="call_0"), DummyOutput(sequence=mock_seq)),
            (_make_submit_msg(), DummyOutput(sequence=mock_seq)),
        ]
        client = DummyClient(responses)

        with _patch_execute_tool() as mock_et:
            mock_et.side_effect = [
                ("tool output", False, None),
                ("Submitted.", True, "train"),
            ]
            result = asyncio.run(
                _make_agent(client).run(
                    [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "task"},
                    ]
                )
            )

        assert result.trajectory is not None
        assert len(result.trajectory.sequences) == 2


class TestMessagesHistory:
    """O. Messages history is correct."""

    def test_message_ordering(self):
        responses = [
            (_make_tool_call_msg(call_id="call_0"), DummyOutput()),
            (_make_submit_msg(call_id="call_1"), DummyOutput()),
        ]
        client = DummyClient(responses)

        with _patch_execute_tool() as mock_et:
            mock_et.side_effect = [
                ("tool output", False, None),
                ("Submitted.", True, "train"),
            ]
            initial_msgs = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
            ]
            result = asyncio.run(_make_agent(client).run(initial_msgs))

        msgs = result.messages
        # Expected: system, user, assistant, tool, assistant, tool
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "tool"
        assert msgs[4]["role"] == "assistant"
        assert msgs[5]["role"] == "tool"

        # Check tool_call_id matching
        assert msgs[3]["tool_call_id"] == "call_0"
        assert msgs[5]["tool_call_id"] == "call_1"


class TestRolloutTimeout:
    """Rollout timeout terminates the loop."""

    def test_timeout(self):
        responses = [(_make_tool_call_msg(call_id="call_0"), DummyOutput())]
        client = DummyClient(responses)

        # Set rollout_timeout to 0 so it immediately triggers
        result = asyncio.run(
            _make_agent(client, rollout_timeout=0.0).run(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                ]
            )
        )

        assert result.metrics["termination_reason"] == "rollout_timeout"
