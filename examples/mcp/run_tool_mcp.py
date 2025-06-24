import asyncio
import os
import sys

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

from rllm.agents.system_prompts import SEARCH_SYSTEM_PROMPT
from rllm.agents.tool_agent import MCPToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.mcp_env import MCPConnectionManager, MCPEnvironment
from rllm.rewards.reward_fn import search_reward_fn
from rllm.utils import save_trajectories

load_dotenv()


def load_hotpotqa_data(test_size=50):
    if DatasetRegistry.dataset_exists("hotpotqa", "test"):
        test_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")
        return test_dataset.get_data()

    print("Loading HotpotQA dataset...")
    hotpot_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)
    hotpot_val = hotpot_dataset["validation"]

    hotpot_val_subset = hotpot_val.select(range(min(test_size, len(hotpot_val))))

    def process_hotpot_example(example, idx):
        question = example["question"]
        ground_truth = example["answer"]

        return {
            "question": question,
            "ground_truth": ground_truth,
            "data_source": "hotpotqa",
            "uid": f"hotpot_{example.get('id', idx)}",
            "question_type": example.get("type", "bridge"),
            "level": example.get("level", "medium"),
            "task_info": {
                "question": question,
                "ground_truth": ground_truth,
                "data_source": "hotpotqa",
            },
        }

    print("Processing HotpotQA validation data...")
    hotpot_val_processed = [process_hotpot_example(example, idx) for idx, example in enumerate(hotpot_val_subset)]

    print(f"Processed {len(hotpot_val_processed)} HotpotQA examples")

    test_dataset = DatasetRegistry.register_dataset("hotpotqa", hotpot_val_processed, "test")

    return test_dataset.get_data()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python run_tool_mcp.py <tavily_api_key>")
        print("This will run HotpotQA evaluation using Tavily MCP server")
        sys.exit(1)

    tavily_api_key = sys.argv[1]

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    n_parallel_agents = 4
    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    mcp_server_command = "npx"
    mcp_server_args = ["-y", "tavily-mcp@0.2.4"]
    mcp_server_env = {"TAVILY_API_KEY": tavily_api_key}

    temp_manager = MCPConnectionManager(mcp_server_command, mcp_server_args, mcp_server_env)
    temp_manager.start()
    try:
        mcp_tool_map = temp_manager.tool_map
        print(f"Available tools: {list(mcp_tool_map.keys())}")
    finally:
        temp_manager.stop()

    agent_args = {"parser_name": "qwen", "system_prompt": SEARCH_SYSTEM_PROMPT, "tool_map": mcp_tool_map}

    env_args = {
        "mcp_server_command": mcp_server_command,
        "mcp_server_args": mcp_server_args,
        "mcp_server_env": mcp_server_env,
        "reward_fn": search_reward_fn,
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=MCPToolAgent,
        env_class=MCPEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={"base_url": "http://localhost:30000/v1", "api_key": "None"},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=16384,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
    )

    test_data = load_hotpotqa_data(test_size=10)  # Start with 10 for testing

    tasks = []
    for item in test_data:
        task = {"question": item["question"], "ground_truth": item["ground_truth"], "data_source": "hotpotqa"}
        tasks.append(task)

    print(f"Running evaluation on {len(tasks)} HotpotQA tasks...")

    try:
        results = await engine.execute_tasks(tasks)

        save_trajectories(results, save_dir="./trajectories/mcp_tavily", filename="trajectories.pt")

    finally:
        MCPEnvironment.cleanup_global_resources()


if __name__ == "__main__":
    asyncio.run(main())
