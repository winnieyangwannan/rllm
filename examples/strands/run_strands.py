import asyncio
import os
import logging
from dotenv import load_dotenv, find_dotenv

from transformers import AutoTokenizer
from rllm.engine.rollout import OpenAIEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent
from strands_tools import calculator, http_request, file_read, python_repl
from strands_tools.browser import LocalChromiumBrowser

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

    # Initialize browser tool (graceful fallback if runtime not installed)
    try:
        browser = LocalChromiumBrowser()
        browser_tool = browser.browser
    except Exception as e:
        logging.warning(f"Browser disabled: {e}")
        browser = None
        browser_tool = None

    # Prepare tool set with all available tools
    # Include all available tools - let strands handle the tool specifications
    tools = [
        calculator.calculator,  # Modern @tool decorator format
        http_request,           # Native strands format with TOOL_SPEC
        file_read,             # Native strands format with TOOL_SPEC  
        python_repl,           # Native strands format with TOOL_SPEC
        google_search          # Custom tool
    ]
    # Disable browser tool for now
    # if browser_tool:
    #     tools.append(browser_tool)

    # System prompt with all available tools
    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful agent with access to multiple tools. Use tools when beneficial and keep answers concise. Available tools: calculator for math calculations, http_request for API calls, file_read for reading files, python_repl for code execution, and google_search for current information. Prefer google_search first for information queries.",
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
