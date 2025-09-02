import asyncio
import os
import logging
from dotenv import load_dotenv, find_dotenv

from transformers import AutoTokenizer
from rllm.engine.rollout import OpenAIEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent
from strands_tools.calculator import calculator
from gsearch_tool_wrapped import google_search

# Disable OpenTelemetry SDK
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# Disable OpenTelemetry error logs
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("strands.telemetry").setLevel(logging.CRITICAL)


async def run_strands_agent(rollout_engine):
    """Example using StrandsAgent with trajectory tracking."""

    # Create RLLMModel
    model = RLLMModel(rollout_engine=rollout_engine, model_id="Qwen/Qwen3-0.6B")

    # Prepare minimal tool set
    tools = [calculator, google_search]

    # Simple system prompt
    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful agent. Use tools when beneficial, and keep answers concise. Use the google_search tool to search for current information.",
    )

    # Create StrandsAgent with trajectory tracking and tools
    agent = StrandsAgent(model=model, tools=tools, system_prompt=system_prompt)

    # Reset trajectory for new task
    task = os.getenv(
        "STRANDS_TASK",
        "who won the 2025 US Tennis Open?",
    )
    agent.reset_trajectory(task=task)

    print(f"📝 Task: {task}")

    # Run the agent
    result = agent(task)
    
    # # Display the result
    # if hasattr(result, 'message'):
    #     if isinstance(result.message, dict):
    #         content = result.message.get('content', [])
    #     elif hasattr(result.message, 'content'):
    #         content = result.message.content
    #     else:
    #         content = []
        
    #     # Extract text content
    #     if isinstance(content, list):
    #         for event in content:
    #             if isinstance(event, dict) and 'text' in event:
    #                 print(f"🤖 Response: {event['text']}")
    #                 break
    
    print(f"\n✅ Final result: {repr(result)}")


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

    await run_strands_agent(rollout_engine)


if __name__ == "__main__":
    asyncio.run(main())
