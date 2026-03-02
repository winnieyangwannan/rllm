"""Relevance evaluator — checks if the agent's answer mentions the expected cuisine."""

from __future__ import annotations

from rllm.experimental.eval.types import EvalOutput, Signal, _extract_agent_answer
from rllm.types import Episode


class RelevanceEvaluator:
    """Checks if the recommendation mentions the expected cuisine."""

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        answer = _extract_agent_answer(episode)
        expected_cuisine = task.get("cuisine", "")

        is_relevant = expected_cuisine.lower() in answer.lower() if expected_cuisine else False

        return EvalOutput(
            reward=1.0 if is_relevant else 0.0,
            is_correct=is_relevant,
            signals=[Signal(name="relevance", value=float(is_relevant))],
        )
