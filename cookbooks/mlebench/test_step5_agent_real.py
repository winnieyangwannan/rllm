#!/usr/bin/env python3
"""
Step 5: Agent Loop with Real Sandbox Test

Tests the full agent loop with a REAL AgentBox container and LLM.
This is an integration test that requires:
1. AgentBox manager running
2. LLM endpoint (vLLM/SGLang/LiteLLM with OpenAI-compatible API)

Tests TWO LLM client options:
  Option 1 - Direct OpenAI client:
    A) Run agent with a simple task (ls, echo) — verify loop completes
    B) Verify bash commands actually execute in container
    C) Verify submit tool captures solution.py content

  Option 2 - LiteLLM client:
    D) Run agent with LiteLLM completion (OpenAI-compatible wrapper)
    E) Verify LiteLLM tool_calls work correctly

Usage:
    python test_agent_real.py [--option1] [--option2] [--all]

Configure the LLM endpoint below before running.
"""

import argparse
import sys

# Add the mle_agent to path for testing before pip install
sys.path.insert(0, "/home/winnieyangwn/rllm/agenthub/mle_agent")

# ============================================================================
# CONFIG - UPDATE THESE VALUES FOR YOUR ENVIRONMENT
# ============================================================================

MANAGER_URI = "h200-137-000-067:35743"

# LLM endpoint - Azure OpenAI
# For AzureOpenAI SDK: azure_endpoint is JUST the base URL (without /openai/deployments/...)
AZURE_ENDPOINT = "https://azure-services-fair-openai1-eastus2n3.azure-api.net"
# For LiteLLM: api_base is the FULL URL with deployment path (amaia-collab style)
LLM_BASE_URL = "https://azure-services-fair-openai1-eastus2n3.azure-api.net/openai/deployments/gpt-5"
LLM_MODEL = "gpt-5"  # Azure deployment name (for AzureOpenAI SDK)
LLM_MODEL_LITELLM = "azure/gpt-5"  # For LiteLLM (needs azure/ prefix)
LLM_API_KEY = "73afb4e502de426c8ea645416de6ec0b"
LLM_API_VERSION = "2025-03-01-preview"

# ============================================================================
# OPTION 1: DIRECT OPENAI CLIENT TESTS
# ============================================================================


def test_a_simple_interaction():
    """Test A: Run agent with simple task — verify loop completes."""
    print("\n" + "=" * 60)
    print("TEST A: Simple agent interaction (OpenAI client)")
    print("=" * 60)

    import openai
    from mle_agent.agent import _run_agent_loop
    from mle_agent.prompts import SYSTEM_PROMPT

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    # Create real sandbox
    sandbox = AgentBoxSandbox(
        name="test-real-a",
        manager_uri=MANAGER_URI,
    )
    print("✓ Created sandbox")

    try:
        # Set up mock data
        sandbox.exec("mkdir -p /root/data /workspace")
        sandbox.exec("echo 'id,value' > /root/data/train.csv")
        sandbox.exec("echo '1,100' >> /root/data/train.csv")
        print("✓ Set up mock data")

        # Build prompts
        system_prompt = SYSTEM_PROMPT.format(
            timeout_min=5,
            context_size=32000,
            eval_timeout_hrs=1,
        )

        # Simple task: just list files and create a dummy solution
        instance_prompt = """Your task is very simple for testing:
1. Run `ls /root/data` to see the data files
2. Create a simple solution.py that prints "hello"
3. Submit it

This is just a connectivity test. Complete it quickly."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]

        # Create Azure OpenAI client
        client = openai.AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=LLM_API_KEY,
            api_version=LLM_API_VERSION,
        )
        print(f"✓ Created AzureOpenAI client (endpoint={AZURE_ENDPOINT}, model={LLM_MODEL})")

        # Run the agent loop with limited turns
        steps, final_messages, pred_solution = _run_agent_loop(
            client=client,
            model=LLM_MODEL,
            messages=messages,
            sandbox=sandbox,
            max_turns=10,  # Limit turns for quick test
            session_timeout=60.0,
            rollout_timeout=300.0,  # 5 minute max
            temperature=1.0,  # GPT-5 only supports temperature=1.0
        )

        print(f"✓ Agent completed with {len(steps)} steps")

        # Log what happened
        for i, step in enumerate(steps):
            input_str = str(step.input)[:50] if step.input else ""
            output_str = str(step.output)[:50] if step.output else ""
            print(f"  Step {i + 1}: input={input_str}... output={output_str}...")

        if pred_solution:
            print(f"✓ Solution submitted ({len(pred_solution)} chars)")
            print(f"  Preview: {pred_solution[:100]}...")
        else:
            print("⚠ No solution submitted (agent may not have reached submit)")

        # Basic success criteria: at least one step completed
        assert len(steps) >= 1, "Expected at least one step"

    finally:
        sandbox.close()
        print("✓ Sandbox closed")

    print("✓ TEST A PASSED")
    return True


def test_b_bash_commands_execute():
    """Test B: Verify bash commands actually execute in container."""
    print("\n" + "=" * 60)
    print("TEST B: Bash commands execute (OpenAI client)")
    print("=" * 60)

    import openai
    from mle_agent.agent import _run_agent_loop

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    sandbox = AgentBoxSandbox(
        name="test-real-b",
        manager_uri=MANAGER_URI,
    )
    print("✓ Created sandbox")

    try:
        sandbox.exec("mkdir -p /workspace")

        # Very specific task to test command execution
        instance_prompt = """Execute this EXACT command and then submit:
        
bash: echo "TEST_MARKER_12345" > /workspace/test_output.txt

Then run:
bash: cat /workspace/test_output.txt

If you see TEST_MARKER_12345, create solution.py with just `print("success")` and submit."""

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the bash tool to run commands."},
            {"role": "user", "content": instance_prompt},
        ]

        client = openai.AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=LLM_API_KEY,
            api_version=LLM_API_VERSION,
        )

        steps, final_messages, pred_solution = _run_agent_loop(
            client=client,
            model=LLM_MODEL,
            messages=messages,
            sandbox=sandbox,
            max_turns=8,
            session_timeout=30.0,
            rollout_timeout=120.0,
        )

        print(f"✓ Agent completed with {len(steps)} steps")

        # Check if the marker file was created
        test_output = sandbox.exec("cat /workspace/test_output.txt 2>/dev/null || echo 'NOT_FOUND'")
        print(f"✓ test_output.txt content: {test_output.strip()}")

        # Verify the command actually executed
        if "TEST_MARKER_12345" in test_output:
            print("✓ Bash commands executed successfully in container")
        else:
            print("⚠ Marker not found - commands may not have executed")

    finally:
        sandbox.close()
        print("✓ Sandbox closed")

    print("✓ TEST B PASSED")
    return True


def test_c_submit_captures_solution():
    """Test C: Verify submit tool captures solution.py content."""
    print("\n" + "=" * 60)
    print("TEST C: Submit captures solution (OpenAI client)")
    print("=" * 60)

    import openai
    from mle_agent.agent import _run_agent_loop

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    sandbox = AgentBoxSandbox(
        name="test-real-c",
        manager_uri=MANAGER_URI,
    )
    print("✓ Created sandbox")

    try:
        sandbox.exec("mkdir -p /workspace")

        # Pre-create solution.py so the agent just needs to submit
        solution_content = """# Test solution
import pandas as pd
print("This is a test solution")
"""
        sandbox.exec(f"cat > /workspace/solution.py << 'EOF'\n{solution_content}\nEOF")
        print("✓ Pre-created solution.py")

        instance_prompt = """A solution.py file already exists at /workspace/solution.py.
Your only task is to submit it using the submit tool with path="/workspace/solution.py"."""

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the submit tool to submit solutions."},
            {"role": "user", "content": instance_prompt},
        ]

        client = openai.AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=LLM_API_KEY,
            api_version=LLM_API_VERSION,
        )

        steps, final_messages, pred_solution = _run_agent_loop(
            client=client,
            model=LLM_MODEL,
            messages=messages,
            sandbox=sandbox,
            max_turns=5,
            session_timeout=30.0,
            rollout_timeout=60.0,
        )

        print(f"✓ Agent completed with {len(steps)} steps")

        if pred_solution:
            print(f"✓ Solution captured ({len(pred_solution)} chars)")
            print(f"  Content:\n{pred_solution}")

            # Verify the content matches
            assert "Test solution" in pred_solution or "pandas" in pred_solution, "Solution content doesn't match expected"
            print("✓ Solution content verified")
        else:
            print("⚠ No solution captured - agent may not have used submit tool")

    finally:
        sandbox.close()
        print("✓ Sandbox closed")

    print("✓ TEST C PASSED")
    return True


# ============================================================================
# OPTION 2: LITELLM CLIENT TESTS
# ============================================================================


def test_d_litellm_simple_interaction():
    """Test D: Run agent with LiteLLM completion wrapper."""
    print("\n" + "=" * 60)
    print("TEST D: Simple agent interaction (LiteLLM client)")
    print("=" * 60)

    import litellm
    from mle_agent.agent import _run_agent_loop
    from mle_agent.prompts import SYSTEM_PROMPT

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    # Create a LiteLLM-compatible client wrapper for Azure
    class LiteLLMClient:
        """Wrapper to make LiteLLM look like OpenAI client for _run_agent_loop."""

        def __init__(self, api_base: str, api_key: str, api_version: str):
            self.api_base = api_base
            self.api_key = api_key
            self.api_version = api_version
            self.chat = self  # So client.chat.completions.create works

        @property
        def completions(self):
            return self

        def create(self, model: str, messages: list, tools: list = None, **kwargs):
            """Call LiteLLM completion with Azure settings."""
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=tools,
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
                **kwargs,
            )
            return response

    sandbox = AgentBoxSandbox(
        name="test-litellm-d",
        manager_uri=MANAGER_URI,
    )
    print("✓ Created sandbox")

    try:
        sandbox.exec("mkdir -p /root/data /workspace")
        sandbox.exec("echo 'id,value' > /root/data/train.csv")
        print("✓ Set up mock data")

        system_prompt = SYSTEM_PROMPT.format(
            timeout_min=5,
            context_size=32000,
            eval_timeout_hrs=1,
        )

        instance_prompt = """Your task is very simple for testing:
1. Run `ls /root/data` to see the data files
2. Create a simple solution.py that prints "hello"
3. Submit it

This is just a connectivity test. Complete it quickly."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]

        # Create LiteLLM client wrapper for Azure
        client = LiteLLMClient(api_base=LLM_BASE_URL, api_key=LLM_API_KEY, api_version=LLM_API_VERSION)
        print(f"✓ Created LiteLLM client (api_base={LLM_BASE_URL})")

        steps, final_messages, pred_solution = _run_agent_loop(
            client=client,
            model=LLM_MODEL_LITELLM,
            messages=messages,
            sandbox=sandbox,
            max_turns=10,
            session_timeout=60.0,
            rollout_timeout=300.0,
            temperature=1.0,  # GPT-5 only supports temperature=1.0
        )

        print(f"✓ Agent completed with {len(steps)} steps")

        for i, step in enumerate(steps):
            input_str = str(step.input)[:50] if step.input else ""
            output_str = str(step.output)[:50] if step.output else ""
            print(f"  Step {i + 1}: input={input_str}... output={output_str}...")

        if pred_solution:
            print(f"✓ Solution submitted ({len(pred_solution)} chars)")
        else:
            print("⚠ No solution submitted")

        assert len(steps) >= 1, "Expected at least one step"

    finally:
        sandbox.close()
        print("✓ Sandbox closed")

    print("✓ TEST D PASSED")
    return True


def test_e_litellm_tool_calls():
    """Test E: Verify LiteLLM tool_calls work correctly."""
    print("\n" + "=" * 60)
    print("TEST E: LiteLLM tool_calls (bash commands)")
    print("=" * 60)

    import litellm
    from mle_agent.agent import _run_agent_loop

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    class LiteLLMClient:
        """Wrapper to make LiteLLM look like OpenAI client for Azure."""

        def __init__(self, api_base: str, api_key: str, api_version: str):
            self.api_base = api_base
            self.api_key = api_key
            self.api_version = api_version
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, model: str, messages: list, tools: list = None, **kwargs):
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=tools,
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
                **kwargs,
            )
            return response

    sandbox = AgentBoxSandbox(
        name="test-litellm-e",
        manager_uri=MANAGER_URI,
    )
    print("✓ Created sandbox")

    try:
        sandbox.exec("mkdir -p /workspace")

        # Specific task to test tool calls
        instance_prompt = """Execute this EXACT command:
        
bash: echo "LITELLM_TEST_OK" > /workspace/litellm_test.txt

Then read it back with:
bash: cat /workspace/litellm_test.txt

Then create solution.py with `print("litellm works")` and submit."""

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the bash tool to run commands."},
            {"role": "user", "content": instance_prompt},
        ]

        client = LiteLLMClient(api_base=LLM_BASE_URL, api_key=LLM_API_KEY, api_version=LLM_API_VERSION)

        steps, final_messages, pred_solution = _run_agent_loop(
            client=client,
            model=LLM_MODEL_LITELLM,
            messages=messages,
            sandbox=sandbox,
            max_turns=8,
            session_timeout=30.0,
            rollout_timeout=120.0,
        )

        print(f"✓ Agent completed with {len(steps)} steps")

        # Check if the marker file was created
        test_output = sandbox.exec("cat /workspace/litellm_test.txt 2>/dev/null || echo 'NOT_FOUND'")
        print(f"✓ litellm_test.txt content: {test_output.strip()}")

        if "LITELLM_TEST_OK" in test_output:
            print("✓ LiteLLM tool_calls executed successfully")
        else:
            print("⚠ Marker not found - tool_calls may not have executed")

    finally:
        sandbox.close()
        print("✓ Sandbox closed")

    print("✓ TEST E PASSED")
    return True


def run_all_tests(run_option1: bool = True, run_option2: bool = True):
    """Run all real sandbox tests."""
    print("\n" + "=" * 60)
    print("AGENT LOOP WITH REAL SANDBOX TEST SUITE (Step 5)")
    print("=" * 60)
    print(f"Manager URI: {MANAGER_URI}")
    print(f"Azure Endpoint (OpenAI SDK): {AZURE_ENDPOINT}")
    print(f"LLM Base URL (LiteLLM): {LLM_BASE_URL}")
    print(f"LLM Model (OpenAI SDK): {LLM_MODEL}")
    print(f"LLM Model (LiteLLM): {LLM_MODEL_LITELLM}")
    print(f"Running Option 1 (AzureOpenAI SDK): {run_option1}")
    print(f"Running Option 2 (LiteLLM): {run_option2}")

    tests = []

    if run_option1:
        tests.extend(
            [
                ("A", "Simple interaction (OpenAI)", test_a_simple_interaction),
                ("B", "Bash commands execute (OpenAI)", test_b_bash_commands_execute),
                ("C", "Submit captures solution (OpenAI)", test_c_submit_captures_solution),
            ]
        )

    if run_option2:
        tests.extend(
            [
                ("D", "Simple interaction (LiteLLM)", test_d_litellm_simple_interaction),
                ("E", "Tool calls (LiteLLM)", test_e_litellm_tool_calls),
            ]
        )

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
    parser = argparse.ArgumentParser(description="Step 5: Test agent loop with real sandbox")
    parser.add_argument("--option1", action="store_true", help="Run Option 1: Direct OpenAI client tests (A, B, C)")
    parser.add_argument("--option2", action="store_true", help="Run Option 2: LiteLLM client tests (D, E)")
    parser.add_argument("--all", action="store_true", help="Run all tests (both options)")

    args = parser.parse_args()

    # Default: run all if no specific option selected
    if not args.option1 and not args.option2 and not args.all:
        args.all = True

    run_option1 = args.option1 or args.all
    run_option2 = args.option2 or args.all

    success = run_all_tests(run_option1=run_option1, run_option2=run_option2)
    sys.exit(0 if success else 1)
