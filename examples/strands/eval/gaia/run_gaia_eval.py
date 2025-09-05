"""Main script for running Gaia dataset evaluation using StrandsAgent.

This script evaluates the Gaia dataset using the existing Strands + RLLM integration.
"""

import asyncio
import os
import logging
import argparse
from dotenv import load_dotenv, find_dotenv

from transformers import AutoTokenizer
from rllm.engine.rollout import OpenAIEngine
import sys
import os
# Add parent directory to path to import tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calculator_tool import calculator
from gsearch_tool_wrapped import google_search

# Import strands_tools for additional capabilities
from strands_tools import http_request, file_read, python_repl

from gaia_evaluator import GaiaEvaluator

# Disable OpenTelemetry SDK
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# Disable OpenTelemetry error logs
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("strands.telemetry").setLevel(logging.CRITICAL)


def setup_rollout_engine():
    """Setup the rollout engine based on environment variables."""
    load_dotenv(find_dotenv())
    
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

    # Suppress tokenizer warning for cleaner output
    import warnings
    warnings.filterwarnings("ignore", message="No tokenizer provided")
    
    rollout_engine = OpenAIEngine(
        model=model_id,
        tokenizer=None,
        base_url=base_url,
        api_key=api_key,
        sampling_params={"temperature": 0.7, "top_p": 0.95, "max_tokens": 2048},
    )
    
    return rollout_engine


def run_gaia_evaluation(args):
    """Run the Gaia evaluation."""
    print("🚀 Starting Gaia Dataset Evaluation")
    print("=" * 50)
    
    # Setup rollout engine
    print("📡 Setting up rollout engine...")
    rollout_engine = setup_rollout_engine()
    
    # Prepare tools with all available capabilities
    tools = [
        calculator,           # Custom calculator tool
        http_request,         # Native strands format with TOOL_SPEC
        file_read,           # Native strands format with TOOL_SPEC  
        python_repl,         # Native strands format with TOOL_SPEC
        google_search        # Custom search tool
    ]
    print(f"🔧 Loaded {len(tools)} tools: {[tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in tools]}")
    
    # Custom system prompt for Gaia tasks with all available tools
    system_prompt = (
"""You are an expert agent solving web-based tasks from the Gaia dataset. 
These tasks often require searching for current information, analyzing data, 
and providing accurate answers. Use the available tools when needed:
- calculator: for mathematical calculations
- http_request: for API calls and web requests
- file_read: for reading and analyzing files
- python_repl: for executing Python code
- google_search: for current information and web searches

Think step by step and use tools as needed. 
Explain your reasoning in the middle steps if it helps you decide.

IMPORTANT:
At the end of your reasoning, output the final answer **ONLY ONCE**. Do not include any explanation with the Final Answer.
"""    
)
    
    # Initialize evaluator
    print("🎯 Initializing Gaia evaluator...")
    evaluator = GaiaEvaluator(
        rollout_engine=rollout_engine,
        tools=tools,
        system_prompt=system_prompt,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset not found at: {args.dataset_path}")
        print("💡 Please run the download script first:")
        print("   python scripts/data/download_gaia.py")
        return
    
    # Run evaluation
    print(f"📊 Evaluating dataset: {args.dataset_path}")
    results = evaluator.evaluate_dataset(args.dataset_path)
    
    # Generate and save results
    print("💾 Saving evaluation results...")
    json_path, csv_path = evaluator.save_results(args.output_filename)
    
    # Print summary
    report = evaluator.generate_report()
    summary = report["summary"]
    
    print("\n" + "=" * 50)
    print("📈 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Correct answers: {summary['correct_samples']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Exact match rate: {summary['exact_match_rate']:.2%}")
    print(f"Average F1 score: {summary['average_f1_score']:.3f}")
    print(f"Average execution time: {summary['average_execution_time']:.2f}s")
    print(f"Tool usage: {report['tool_usage']}")
    
    if report['error_count'] > 0:
        print(f"⚠️  Errors encountered: {report['error_count']}")
    
    print(f"\n📁 Results saved to:")
    print(f"   JSON: {json_path}")
    print(f"   CSV: {csv_path}")
    
    print("\n✅ Gaia evaluation completed!")


async def main():
    """Main async function."""
    parser = argparse.ArgumentParser(description="Run Gaia dataset evaluation using StrandsAgent")
    parser.add_argument(
        "--dataset_path", 
        default="../../../../rllm/data/train/web/gaia.json",
        help="Path to the Gaia dataset JSON file"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output_dir", 
        default="outputs/gaia_eval",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--output_filename", 
        default=None,
        help="Base filename for output files (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    run_gaia_evaluation(args)


if __name__ == "__main__":
    asyncio.run(main())
