import asyncio
import os

from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer

from rllm.engine.rollout.openai_engine import OpenAIEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent

os.environ.setdefault("OTEL_SDK_DISABLED", "true")   # disable OTEL SDK


async def run_strands_agent(rollout_engine):
    """Example using StrandsAgent with trajectory tracking."""

    # Create RLLMModel
    model = RLLMModel(rollout_engine=rollout_engine, model_id="Qwen/Qwen3-0.6B")

    # Create StrandsAgent with trajectory tracking
    agent = StrandsAgent(model=model)

    # Reset trajectory for new task
    task = "Explain the concept of machine learning in simple terms"
    agent.reset_trajectory(task=task)

    print(f"📝 Task: {task}")

    # Run the agent
    result = agent("Explain the concept of machine learning in simple terms, using an analogy that a child could understand.")
    print(result)


async def main():
    load_dotenv(find_dotenv())

    # Tokenizer/model
    model_name = os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-0.6B")
    # Tokenizer is optional; omit it to force chat.completions endpoint for chat models
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Provider selection (Together or OpenAI-compatible)
    together_api_key = os.getenv("TOGETHER_AI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if together_api_key:
        base_url = "https://api.together.xyz/v1"
        api_key = together_api_key
        model_id = os.getenv("TOGETHER_AI_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo")
    elif openai_api_key:
        base_url = os.getenv("OPENAI_BASE_URL")  # optional proxy/base
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
