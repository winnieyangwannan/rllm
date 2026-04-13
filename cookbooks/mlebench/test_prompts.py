#!/usr/bin/env python3
"""
Step 3: Prompts & Tool Schemas Test

Tests that prompts render correctly and tool schemas are valid for OpenAI API.

Tests:
  A) SYSTEM_PROMPT.format(...) renders correctly with sample values
  B) INSTANCE_PROMPT.format(...) renders correctly with sample task data
  C) Tool schemas pass OpenAI API validation (mock test - no LLM call needed)
  D) DATA_INFO_COMMAND and CHECK_SUBMISSION_COMMAND run in container

Usage:
    python test_prompts.py
"""

import sys

# Add the mle_bench_agent to path for testing before pip install
sys.path.insert(0, "/home/winnieyangwn/rllm/agenthub/mle_bench_agent")

# ============================================================================
# CONFIG
# ============================================================================

MANAGER_URI = "h200-137-000-067:35743"

# ============================================================================
# TEST IMPLEMENTATION
# ============================================================================


def test_a_system_prompt_format():
    """Test A: SYSTEM_PROMPT.format() renders correctly."""
    print("\n" + "=" * 60)
    print("TEST A: System prompt formatting")
    print("=" * 60)

    from mle_bench_agent.prompts import SYSTEM_PROMPT

    rendered = SYSTEM_PROMPT.format(
        timeout_min=6,
        context_size=131072,
        max_turns=128,
        eval_timeout_hrs=5,
    )

    print(f"✓ System prompt rendered ({len(rendered)} chars)")

    # Verify key content is present
    assert "AUTONOMOUS ML engineering agent" in rendered
    assert "6 minutes" in rendered or "6" in rendered
    assert "131072" in rendered
    assert "5 hours" in rendered or "5" in rendered
    assert "/root/data/" in rendered
    assert "/workspace" in rendered

    print("✓ All expected content found")
    print(f"\nPreview:\n{rendered[:500]}...")

    print("✓ TEST A PASSED")
    return True


def test_b_instance_prompt_format():
    """Test B: INSTANCE_PROMPT.format() renders correctly."""
    print("\n" + "=" * 60)
    print("TEST B: Instance prompt formatting")
    print("=" * 60)

    from mle_bench_agent.prompts import INSTANCE_PROMPT

    sample_task = {
        "task_description": "Predict house prices based on features like square footage, bedrooms, etc.",
        "data_info": "=== DATA STRUCTURE ===\ntrain.csv  test.csv  sample_submission.csv\n\n=== CSV ROW COUNTS ===\n1000 train.csv\n500 test.csv",
    }

    rendered = INSTANCE_PROMPT.format(
        task_description=sample_task["task_description"],
        data_info=sample_task["data_info"],
    )

    print(f"✓ Instance prompt rendered ({len(rendered)} chars)")

    # Verify content
    assert "house prices" in rendered
    assert "train.csv" in rendered
    assert "submit tool" in rendered

    print("✓ All expected content found")
    print(f"\nFull prompt:\n{rendered}")

    print("✓ TEST B PASSED")
    return True


def test_c_tool_schemas_valid():
    """Test C: Tool schemas are valid JSON Schema format."""
    print("\n" + "=" * 60)
    print("TEST C: Tool schema validation")
    print("=" * 60)

    from mle_bench_agent.prompts import BASH_TOOL, CHECK_SUBMISSION_TOOL, SUBMIT_TOOL

    tools = [BASH_TOOL, SUBMIT_TOOL, CHECK_SUBMISSION_TOOL]

    for tool in tools:
        # Validate structure
        assert tool["type"] == "function", "Tool type must be 'function'"
        assert "function" in tool, "Tool must have 'function' key"

        func = tool["function"]
        assert "name" in func, "Function must have 'name'"
        assert "description" in func, "Function must have 'description'"
        assert "parameters" in func, "Function must have 'parameters'"

        params = func["parameters"]
        assert params["type"] == "object", "Parameters must be object type"
        assert "properties" in params, "Parameters must have 'properties'"
        assert "required" in params, "Parameters must have 'required'"

        print(f"✓ {func['name']}: valid schema")

    # Verify specific tools
    assert BASH_TOOL["function"]["name"] == "bash"
    assert "command" in BASH_TOOL["function"]["parameters"]["properties"]

    assert SUBMIT_TOOL["function"]["name"] == "submit"
    assert "path" in SUBMIT_TOOL["function"]["parameters"]["properties"]

    assert CHECK_SUBMISSION_TOOL["function"]["name"] == "check_submission_validity"

    print("✓ All tool schemas valid")

    print("✓ TEST C PASSED")
    return True


def test_d_commands_run_in_container():
    """Test D: DATA_INFO_COMMAND and helper commands run in container."""
    print("\n" + "=" * 60)
    print("TEST D: Shell commands run in container")
    print("=" * 60)

    from mle_bench_agent.prompts import CHECK_SUBMISSION_COMMAND, DATA_INFO_COMMAND

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    sandbox = AgentBoxSandbox(
        name="test-prompts-d",
        manager_uri=MANAGER_URI,
    )
    print("✓ Created sandbox")

    try:
        # Create mock data directory structure
        setup_cmds = [
            "mkdir -p /root/data",
            "echo 'id,target' > /root/data/sample_submission.csv",
            "echo '1,0' >> /root/data/sample_submission.csv",
            "echo '2,1' >> /root/data/sample_submission.csv",
            "echo 'id,feature1,target' > /root/data/train.csv",
            "echo '1,0.5,0' >> /root/data/train.csv",
        ]
        for cmd in setup_cmds:
            sandbox.exec(cmd)
        print("✓ Created mock data files")

        # Test DATA_INFO_COMMAND
        data_info_output = sandbox.exec(DATA_INFO_COMMAND)
        print(f"✓ DATA_INFO_COMMAND output:\n{data_info_output}")

        assert "DATA STRUCTURE" in data_info_output
        assert "sample_submission.csv" in data_info_output or "SAMPLE SUBMISSION" in data_info_output

        # Create a mock solution.py for CHECK_SUBMISSION_COMMAND
        solution_code = """
import pandas as pd
df = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
df.to_csv("/workspace/submission.csv", index=False)
print("Solution complete!")
"""
        sandbox.exec(f"mkdir -p /workspace && cat > /workspace/solution.py << 'EOF'\n{solution_code}\nEOF")
        print("✓ Created mock solution.py")

        # Test CHECK_SUBMISSION_COMMAND
        check_output = sandbox.exec(CHECK_SUBMISSION_COMMAND)
        print(f"✓ CHECK_SUBMISSION_COMMAND output:\n{check_output}")

        assert "SUBMISSION HEAD" in check_output or "submission.csv" in check_output.lower()

    finally:
        sandbox.close()
        print("✓ Sandbox closed")

    print("✓ TEST D PASSED")
    return True


def test_e_truncate_output():
    """Test E: truncate_output function works correctly."""
    print("\n" + "=" * 60)
    print("TEST E: Output truncation")
    print("=" * 60)

    from mle_bench_agent.prompts import truncate_output

    # Short output - no truncation
    short = "Hello world"
    assert truncate_output(short) == short
    print("✓ Short output not truncated")

    # Long output - should truncate
    long = "x" * 50000
    truncated = truncate_output(long, max_length=1000)
    assert len(truncated) < len(long)
    assert "TRUNCATED" in truncated
    print(f"✓ Long output truncated: {len(long)} → {len(truncated)} chars")

    print("✓ TEST E PASSED")
    return True


def run_all_tests():
    """Run all prompt tests."""
    print("\n" + "=" * 60)
    print("PROMPTS & TOOL SCHEMAS TEST SUITE (Step 3)")
    print("=" * 60)

    tests = [
        ("A", "System prompt format", test_a_system_prompt_format),
        ("B", "Instance prompt format", test_b_instance_prompt_format),
        ("C", "Tool schema validation", test_c_tool_schemas_valid),
        ("D", "Commands in container", test_d_commands_run_in_container),
        ("E", "Output truncation", test_e_truncate_output),
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
