"""Built-in chain-of-thought reasoning agent flow."""

from __future__ import annotations

import logging

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)

REASONING_SYSTEM_PROMPT = """\
You are an expert problem solver. Think through the problem step by step, \
considering all relevant domains (mathematics, science, humanities, logic, etc.).

Show your reasoning clearly, then provide your final answer after the marker ANSWER:

For example:
[Your step-by-step reasoning here]

ANSWER: 42"""


class ReasoningAgentFlow:
    """General-purpose chain-of-thought reasoning agent."""

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        question = task.get("question", "")

        response_text = ""
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("LLM call failed: %s", e)

        step = Step(input=question, output=response_text, done=True)
        traj = Trajectory(name="solver", steps=[step])
        return Episode(task=task, trajectories=[traj], artifacts={"answer": response_text})


# Singleton instance for registry
reasoning_agent = ReasoningAgentFlow()
