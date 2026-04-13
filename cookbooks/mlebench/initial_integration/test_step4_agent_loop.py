#!/usr/bin/env python3
"""
Step 4: Agent Loop with Mock Sandbox Test

Tests the _run_agent_loop() logic without a real container or GPU.
Uses a MockSandbox that returns canned responses and a MockOpenAI client.

Tests:
  A) Normal flow: bash → bash → submit → loop exits with pred_solution
  B) Format error recovery: LLM returns no tool calls → retry up to 3 times
  C) Rollout timeout: Set rollout_timeout=0.1s, verify loop exits on time
  D) Output truncation: Long bash output gets truncated
  E) Unknown tool: LLM calls nonexistent tool → error message returned
  F) JSON decode error: Malformed function.arguments → graceful fallback

Usage:
    python test_agent_loop.py
"""

import json
import sys
from dataclasses import dataclass
from typing import Any

# Add the mle_agent to path for testing before pip install
sys.path.insert(0, "/home/winnieyangwn/rllm/agenthub/mle_agent")


# ============================================================================
# MOCK CLASSES
# ============================================================================


class MockSandbox:
    """Mock sandbox that returns canned responses for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        """
        Args:
            responses: Dict mapping command substrings to outputs.
                       If a command contains the key, return the value.
        """
        self.responses = responses or {}
        self.executed_commands: list[str] = []

    def exec(self, command: str, timeout: float | None = None) -> str:
        """Return canned response based on command content."""
        self.executed_commands.append(command)

        # Check for matching response
        for key, response in self.responses.items():
            if key in command:
                return response

        # Default responses
        if "cat" in command and "solution.py" in command:
            return "# Mock solution code\nprint('hello')"
        if "mkdir" in command:
            return ""
        if "echo" in command:
            return "output"

        return f"Mock output for: {command[:50]}"


@dataclass
class MockToolCall:
    """Mock OpenAI tool call."""

    id: str
    function: Any


@dataclass
class MockFunction:
    """Mock OpenAI function."""

    name: str
    arguments: str


@dataclass
class MockMessage:
    """Mock OpenAI message."""

    content: str | None
    tool_calls: list[MockToolCall] | None

    def model_dump(self, exclude_none: bool = False) -> dict:
        result = {"role": "assistant"}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return result


@dataclass
class MockChoice:
    """Mock OpenAI choice."""

    message: MockMessage


@dataclass
class MockCompletion:
    """Mock OpenAI completion."""

    choices: list[MockChoice]


class MockOpenAIClient:
    """Mock OpenAI client that returns scripted responses."""

    def __init__(self, responses: list[MockMessage]):
        """
        Args:
            responses: List of MockMessage objects to return in sequence.
        """
        self.responses = responses
        self.call_count = 0
        self.chat = self  # For client.chat.completions.create()
        self.completions = self

    def create(self, **kwargs) -> MockCompletion:
        """Return next scripted response."""
        if self.call_count >= len(self.responses):
            # Return empty response to end loop
            return MockCompletion(
                choices=[
                    MockChoice(
                        message=MockMessage(
                            content="I'm done.",
                            tool_calls=None,
                        )
                    )
                ]
            )

        msg = self.responses[self.call_count]
        self.call_count += 1
        return MockCompletion(choices=[MockChoice(message=msg)])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def make_bash_call(command: str, call_id: str = "call_1") -> MockMessage:
    """Create a mock message with a bash tool call."""
    return MockMessage(
        content=None,
        tool_calls=[
            MockToolCall(
                id=call_id,
                function=MockFunction(
                    name="bash",
                    arguments=json.dumps({"command": command}),
                ),
            )
        ],
    )


def make_submit_call(path: str = "/workspace/solution.py", call_id: str = "call_1") -> MockMessage:
    """Create a mock message with a submit tool call."""
    return MockMessage(
        content=None,
        tool_calls=[
            MockToolCall(
                id=call_id,
                function=MockFunction(
                    name="submit",
                    arguments=json.dumps({"path": path}),
                ),
            )
        ],
    )


def make_text_only_response(text: str) -> MockMessage:
    """Create a mock message with only text (no tool calls)."""
    return MockMessage(content=text, tool_calls=None)


def make_unknown_tool_call(name: str = "unknown_tool", call_id: str = "call_1") -> MockMessage:
    """Create a mock message with an unknown tool call."""
    return MockMessage(
        content=None,
        tool_calls=[
            MockToolCall(
                id=call_id,
                function=MockFunction(
                    name=name,
                    arguments=json.dumps({"arg": "value"}),
                ),
            )
        ],
    )


def make_malformed_args_call(call_id: str = "call_1") -> MockMessage:
    """Create a mock message with malformed JSON in arguments."""
    return MockMessage(
        content=None,
        tool_calls=[
            MockToolCall(
                id=call_id,
                function=MockFunction(
                    name="bash",
                    arguments="this is not valid json {{{",
                ),
            )
        ],
    )


# ============================================================================
# TESTS
# ============================================================================


def test_a_normal_flow():
    """Test A: Normal flow — bash → bash → submit → exits with solution."""
    print("\n" + "=" * 60)
    print("TEST A: Normal flow")
    print("=" * 60)

    from mle_agent.agent import _run_agent_loop

    # Script: LLM calls bash twice, then submits
    mock_responses = [
        make_bash_call("ls /root/data"),
        make_bash_call("python train.py"),
        make_submit_call("/workspace/solution.py"),
    ]
    client = MockOpenAIClient(mock_responses)

    sandbox = MockSandbox(
        {
            "ls": "train.csv test.csv",
            "python": "Training complete!",
            "cat": "import pandas as pd\nprint('solution')",
        }
    )

    messages = [{"role": "user", "content": "Solve the task"}]

    steps, final_messages, pred_solution = _run_agent_loop(
        client=client,
        model="test-model",
        messages=messages,
        sandbox=sandbox,
        max_turns=10,
    )

    print(f"✓ Steps: {len(steps)}")
    print(f"✓ Sandbox commands: {sandbox.executed_commands}")
    print(f"✓ Pred solution: {pred_solution[:50] if pred_solution else None}...")

    assert len(steps) == 3, f"Expected 3 steps, got {len(steps)}"
    assert pred_solution is not None, "Expected pred_solution to be set"
    assert "pandas" in pred_solution or "solution" in pred_solution

    print("✓ TEST A PASSED")
    return True


def test_b_format_error_recovery():
    """Test B: Format error recovery — LLM returns no tool calls, retries."""
    print("\n" + "=" * 60)
    print("TEST B: Format error recovery")
    print("=" * 60)

    from mle_agent.agent import _run_agent_loop

    # Script: 2 text-only responses (format errors), then a bash call, then submit
    mock_responses = [
        make_text_only_response("I think I should..."),  # Format error 1
        make_text_only_response("Let me analyze..."),  # Format error 2
        make_bash_call("echo hello"),  # Finally uses tool
        make_submit_call(),
    ]
    client = MockOpenAIClient(mock_responses)
    sandbox = MockSandbox()

    messages = [{"role": "user", "content": "Solve the task"}]

    steps, final_messages, pred_solution = _run_agent_loop(
        client=client,
        model="test-model",
        messages=messages,
        sandbox=sandbox,
        max_turns=10,
    )

    # Check that format error messages were appended
    format_error_count = sum(1 for m in final_messages if m.get("role") == "user" and "must use one of the available tools" in m.get("content", ""))
    print(f"✓ Format error messages sent: {format_error_count}")
    assert format_error_count == 2, f"Expected 2 format error messages, got {format_error_count}"

    print("✓ TEST B PASSED")
    return True


def test_c_rollout_timeout():
    """Test C: Rollout timeout — loop exits when time budget exceeded."""
    print("\n" + "=" * 60)
    print("TEST C: Rollout timeout")
    print("=" * 60)

    import time

    from mle_agent.agent import _run_agent_loop

    # Create a slow mock client that adds delay per call
    class SlowMockClient(MockOpenAIClient):
        def create(self, **kwargs):
            time.sleep(0.02)  # 20ms per call
            return super().create(**kwargs)

    # Script: Many bash calls (should be interrupted by timeout)
    mock_responses = [make_bash_call(f"echo {i}") for i in range(100)]
    client = SlowMockClient(mock_responses)
    sandbox = MockSandbox()

    messages = [{"role": "user", "content": "Solve the task"}]

    start = time.time()
    steps, final_messages, pred_solution = _run_agent_loop(
        client=client,
        model="test-model",
        messages=messages,
        sandbox=sandbox,
        max_turns=100,
        rollout_timeout=0.15,  # 150ms timeout = ~7 calls max
    )
    elapsed = time.time() - start

    print(f"✓ Elapsed time: {elapsed:.2f}s")
    print(f"✓ Steps completed: {len(steps)}")

    # Should exit early due to timeout, not complete all 100 turns
    assert len(steps) < 20, f"Expected early exit, but got {len(steps)} steps"
    assert pred_solution is None, "Should not have submitted"

    print("✓ TEST C PASSED")
    return True


def test_d_output_truncation():
    """Test D: Long bash output gets truncated."""
    print("\n" + "=" * 60)
    print("TEST D: Output truncation")
    print("=" * 60)

    from mle_agent.agent import _run_agent_loop

    # Create a sandbox that returns very long output
    long_output = "x" * 50000
    sandbox = MockSandbox({"echo": long_output})

    mock_responses = [
        make_bash_call("echo long"),
        make_submit_call(),
    ]
    client = MockOpenAIClient(mock_responses)

    messages = [{"role": "user", "content": "Solve the task"}]

    steps, final_messages, pred_solution = _run_agent_loop(
        client=client,
        model="test-model",
        messages=messages,
        sandbox=sandbox,
        max_turns=10,
    )

    # Find the tool response message with the truncated output
    tool_outputs = [m.get("content", "") for m in final_messages if m.get("role") == "tool"]
    assert len(tool_outputs) > 0, "Expected tool outputs"

    first_output = tool_outputs[0]
    print(f"✓ Output length: {len(first_output)} (was {len(long_output)})")

    assert len(first_output) < len(long_output), "Output should be truncated"
    assert "TRUNCATED" in first_output, "Should contain truncation marker"

    print("✓ TEST D PASSED")
    return True


def test_e_unknown_tool():
    """Test E: Unknown tool call returns error message."""
    print("\n" + "=" * 60)
    print("TEST E: Unknown tool handling")
    print("=" * 60)

    from mle_agent.agent import _run_agent_loop

    mock_responses = [
        make_unknown_tool_call("nonexistent_tool"),
        make_submit_call(),
    ]
    client = MockOpenAIClient(mock_responses)
    sandbox = MockSandbox()

    messages = [{"role": "user", "content": "Solve the task"}]

    steps, final_messages, pred_solution = _run_agent_loop(
        client=client,
        model="test-model",
        messages=messages,
        sandbox=sandbox,
        max_turns=10,
    )

    # Check that error message was returned for unknown tool
    tool_outputs = [m.get("content", "") for m in final_messages if m.get("role") == "tool"]
    assert len(tool_outputs) > 0

    first_output = tool_outputs[0]
    print(f"✓ Tool output: {first_output}")

    assert "Unknown tool" in first_output, "Should report unknown tool error"
    assert "bash" in first_output or "submit" in first_output, "Should list available tools"

    print("✓ TEST E PASSED")
    return True


def test_f_json_decode_error():
    """Test F: Malformed JSON arguments — graceful fallback."""
    print("\n" + "=" * 60)
    print("TEST F: JSON decode error handling")
    print("=" * 60)

    from mle_agent.agent import _run_agent_loop

    mock_responses = [
        make_malformed_args_call(),
        make_submit_call(),
    ]
    client = MockOpenAIClient(mock_responses)
    sandbox = MockSandbox()

    messages = [{"role": "user", "content": "Solve the task"}]

    # Should not raise an exception
    steps, final_messages, pred_solution = _run_agent_loop(
        client=client,
        model="test-model",
        messages=messages,
        sandbox=sandbox,
        max_turns=10,
    )

    print("✓ Completed without exception")
    print(f"✓ Steps: {len(steps)}")

    # The malformed args should have been treated as a raw command string
    assert len(steps) >= 1

    print("✓ TEST F PASSED")
    return True


def run_all_tests():
    """Run all agent loop tests."""
    print("\n" + "=" * 60)
    print("AGENT LOOP TEST SUITE (Step 4)")
    print("=" * 60)

    tests = [
        ("A", "Normal flow", test_a_normal_flow),
        ("B", "Format error recovery", test_b_format_error_recovery),
        ("C", "Rollout timeout", test_c_rollout_timeout),
        ("D", "Output truncation", test_d_output_truncation),
        ("E", "Unknown tool", test_e_unknown_tool),
        ("F", "JSON decode error", test_f_json_decode_error),
    ]

    results = []
    for test_id, test_name, test_fn in tests:
        try:
            success = test_fn()
            results.append((test_id, test_name, success, None))
        except Exception as e:
            import traceback

            print(f"\n✗ TEST {test_id} FAILED: {e}")
            traceback.print_exc()
            results.append((test_id, test_name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, _, success, _ in results if success)
    total = len(results)

    for test_id, test_name, success, error in results:
        status = "✓ PASSED" if success else f"✗ FAILED: {error}"
        print(f"  Test {test_id} ({test_name}): {status}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
