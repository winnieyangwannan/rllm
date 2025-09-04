import asyncio
import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

from transformers import AutoTokenizer
from rllm.engine.rollout import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent
from strands_tools import calculator, http_request, file_read, python_repl
from strands_tools.browser import LocalChromiumBrowser
from strands_workflow import StrandsWorkflow

from gsearch_tool_wrapped import google_search

# Disable OpenTelemetry SDK
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# Disable OpenTelemetry error logs
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("strands.telemetry").setLevel(logging.CRITICAL)


def save_episode_to_json(episode, output_dir="./strands_outputs"):
    """Save the episode to a JSON file with timestamp."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get trajectory data from episode
    trajectory = episode.trajectories[0][1] if episode.trajectories else None
    if not trajectory:
        print("❌ No trajectory found in episode")
        return None
    
    # Convert trajectory to dict format
    trajectory_data = {
        "task": trajectory.task,
        "reward": trajectory.reward,
        "steps": [],
        "tool_calls_summary": [],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_steps": len(trajectory.steps),
            "total_tool_calls": 0,
            "agent_type": "StrandsAgent",
            "episode_id": episode.id
        }
    }
    
    # Convert each step to dict and count tool calls
    tool_call_count = 0
    for i, step in enumerate(trajectory.steps):
        step_data = {
            "step_index": i,
            "observation": step.observation,
            "model_response": step.model_response,
            "action": step.action,
            "reward": step.reward,
            "done": step.done,
            "chat_completions": step.chat_completions
        }
        
        # Add tool call specific information
        if step.action and isinstance(step.action, dict) and step.action.get("type") == "tool_call":
            step_data["step_type"] = "tool_call"
            step_data["tool_name"] = step.action.get("tool_name")
            step_data["tool_args"] = step.action.get("tool_args")
            step_data["tool_result"] = step.action.get("tool_result")
            tool_call_count += 1
        else:
            step_data["step_type"] = "conversation"
        
        trajectory_data["steps"].append(step_data)
    
    # Update tool call count
    trajectory_data["metadata"]["total_tool_calls"] = tool_call_count
    trajectory_data["tool_calls_summary"] = [
        {
            "tool_name": step.action.get("tool_name"),
            "tool_args": step.action.get("tool_args"),
            "tool_result": step.action.get("tool_result")
        }
        for step in trajectory.steps
        if step.action and isinstance(step.action, dict) and step.action.get("type") == "tool_call"
    ]
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"strands_trajectory_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to JSON file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Trajectory saved to: {filepath}")
    
    # Print concise summary
    print(f"📊 Summary: {len(trajectory.steps)} steps, {tool_call_count} tool calls, reward: {trajectory.reward}")
    
    return filepath


async def run_strands_workflow(rollout_engine):
    """Example using StrandsWorkflow with AgentWorkflowEngine."""

    # Create RLLMModel
    model = RLLMModel(rollout_engine=rollout_engine, model_id="Qwen/Qwen3-0.6B")

    # Initialize browser tool (graceful fallback if runtime not installed)
    try:
        browser = LocalChromiumBrowser()
        browser_tool = browser.browser
    except Exception as e:
        logging.warning(f"Browser disabled: {e}")
        browser = None
        browser_tool = None

    # Prepare tool set with all available tools
    tools = [
        calculator.calculator,  # Modern @tool decorator format
        http_request,           # Native strands format with TOOL_SPEC
        # file_read,             # Native strands format with TOOL_SPEC  
        # python_repl,           # Native strands format with TOOL_SPEC
        # google_search          # Custom tool
    ]
    # Disable browser tool for now
    # if browser_tool:
    #     tools.append(browser_tool)

    # System prompt with all available tools
    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful agent with access to multiple tools. Use tools when beneficial and keep answers concise. ",
    )

    # Create AgentWorkflowEngine
    workflow_engine = AgentWorkflowEngine(
        workflow_cls=StrandsWorkflow,
        workflow_args={
            "agent_cls": StrandsAgent,
            "agent_args": {
                "model": model,
                "tools": tools,
                "system_prompt": system_prompt
            }
        },
        rollout_engine=rollout_engine,
        n_parallel_tasks=1  # Single task for now
    )

    # Initialize the workflow pool
    await workflow_engine.initialize_pool()

    # Prepare task
    task = os.getenv(
        "STRANDS_TASK",
        # "A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016? Use http_request to get the information.",
        "What is 303 * 12314 - 123412 * 2? Use calculator to solve this.",
    )
    
    task_dict = {"task": task}
    task_id = "strands_task_001"

    print(f"📝 Task: {task}")
    print(f"🔧 Available tools: {[tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in tools]}")
    print("\n" + "="*80)
    print("🚀 Starting workflow execution...")
    print("="*80)

    # Execute the workflow
    episodes = await workflow_engine.execute_tasks([task_dict], [task_id])
    
    if not episodes:
        print("❌ No episodes returned from workflow")
        return
    
    episode = episodes[0]
    
    print("\n" + "="*80)
    print("✅ Workflow execution completed!")
    print("="*80)
    
    # Save episode to JSON file
    trajectory_file = save_episode_to_json(episode)
    
    # Display concise episode summary
    print(f"\n✅ Episode {episode.id} completed")
    for agent_name, trajectory in episode.trajectories:
        print(f"📊 {agent_name}: {len(trajectory.steps)} steps, reward: {trajectory.reward}")


async def main():
    load_dotenv(find_dotenv())

    # Tokenizer/model
    model_name = os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Provider selection (Together or OpenAI-compatible)
    together_api_key = os.getenv("TOGETHER_AI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if together_api_key:
        base_url = "https://api.together.xyz/v1"
        api_key = together_api_key
        model_id = os.getenv("TOGETHER_AI_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo")
    elif openai_api_key:
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = openai_api_key
        model_id = os.getenv("MODEL_NAME", "gpt-4o")
    else:
        raise ValueError("API key required (TOGETHER_AI_API_KEY or OPENAI_API_KEY)")

    rollout_engine = OpenAIEngine(
        model=model_id,
        tokenizer=None,
        base_url=base_url,
        api_key=api_key,
        sampling_params={"temperature": 0.7, "top_p": 0.95, "max_tokens": 512},
    )

    await run_strands_workflow(rollout_engine)


if __name__ == "__main__":
    asyncio.run(main())
