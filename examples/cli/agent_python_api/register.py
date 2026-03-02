"""Register the concierge agent, evaluator, and dataset via the Python API.

After running this script, the agent and evaluator are discoverable by name
from any process — including the rllm CLI:

    python register.py
    rllm agent list                        # shows "concierge" as registered
    rllm eval concierge --agent concierge --evaluator relevance ...
    rllm train concierge --agent concierge --evaluator relevance --model ...
"""

from __future__ import annotations

from pathlib import Path

from rllm.data import Dataset, DatasetRegistry
from rllm.experimental.eval.agent_loader import register_agent
from rllm.experimental.eval.evaluator_loader import register_evaluator


def main() -> None:
    # Register agent (persists to ~/.rllm/agents.json)
    register_agent("concierge", "concierge_agent.agent:ConciergeAgent")
    print("Registered agent 'concierge'")

    # Register evaluator (persists to ~/.rllm/evaluators.json)
    register_evaluator("relevance", "concierge_agent.evaluator:RelevanceEvaluator")
    print("Registered evaluator 'relevance'")

    # Register dataset (persists to ~/.rllm/datasets/)
    data_path = str(Path(__file__).parent / "data.json")
    ds = Dataset.load_data(data_path)
    for split in ("train", "test"):
        if not DatasetRegistry.dataset_exists("concierge", split=split):
            DatasetRegistry.register_dataset(
                "concierge",
                ds.data,
                split=split,
                category="qa",
                description="Restaurant concierge dataset",
            )
    print(f"Registered dataset 'concierge' ({len(ds)} examples)")

    print("\nDone. You can now use the rllm CLI:")
    print("  rllm agent list")
    print("  rllm eval concierge --agent concierge --evaluator relevance --base-url ... --model ...")
    print("  rllm train concierge --agent concierge --evaluator relevance --model Qwen/Qwen3-8B")


if __name__ == "__main__":
    main()
