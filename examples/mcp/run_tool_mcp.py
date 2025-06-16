import asyncio
import sys

from dotenv import load_dotenv
from transformers import AutoTokenizer

from rllm.agents.tool_agent import MCPToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine
from rllm.environments.tools.mcp_env import MCPConnectionManager, MCPEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.utils import compute_pass_at_k

load_dotenv()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <path_to_mcp_server.py>")
        print("This will run AIME evaluation using the specified MCP server")
        sys.exit(1)

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 8  # Reduced due to MCP connection overhead
    model_name = "Qwen/Qwen3-4B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a temporary connection manager to discover available tools
    temp_manager = MCPConnectionManager(sys.argv[1], [], None)
    temp_manager.start()
    try:
        # Get tool information
        mcp_tool_map = temp_manager.tool_map
    finally:
        temp_manager.stop()
    
    agent_args = {
        "parser_name": "qwen",
        "system_prompt": "You are a math assistant that can use tools to solve math problems. Use the available tools to help solve the problem step by step.",
        "tool_map": mcp_tool_map
    }
    
    env_args = {
        "mcp_server_command": sys.argv[1],
        "mcp_server_args": [],
        "mcp_server_env": None,
        "reward_fn": math_reward_fn,
    }

    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "model": model_name
    }

    engine = AsyncAgentExecutionEngine(
        agent_class=MCPToolAgent,
        env_class=MCPEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai", 
        rollout_engine_args={"base_url": "http://localhost:30000/v1", "api_key": "None"},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=16384,
        max_prompt_length=2048,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
    tasks = test_dataset.repeat(n=1)  # repeat to evaluate pass@k

    # Limit to first few problems for testing
    tasks = tasks[:1]  # Remove this line to run on full dataset
    
    try:
        results = await engine.execute_tasks(tasks)
        print(results)
        compute_pass_at_k(results)
    finally:
        # Clean up global resources
        MCPEnvironment.cleanup_global_resources()


if __name__ == "__main__":
    asyncio.run(main()) 