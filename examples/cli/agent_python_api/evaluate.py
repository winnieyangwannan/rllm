"""Evaluate the concierge agent using the rLLM Python API.

Assumes register.py has been run first.

Usage:
    python evaluate.py --base-url http://localhost:8000/v1 --model gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio

from rllm.data import DatasetRegistry
from rllm.experimental.eval.agent_loader import load_agent
from rllm.experimental.eval.evaluator_loader import load_evaluator
from rllm.experimental.eval.runner import EvalRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the concierge agent")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples")
    args = parser.parse_args()

    # Load by name — works because register.py persisted them to ~/.rllm/
    dataset = DatasetRegistry.load_dataset("concierge", "test")
    agent = load_agent("concierge")
    evaluator = load_evaluator("relevance")

    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    runner = EvalRunner(base_url=args.base_url, model=args.model)
    result = asyncio.run(runner.run(dataset, agent, evaluator, agent_name="concierge"))
    print(result.summary_table())


if __name__ == "__main__":
    main()
