"""Built-in instruction-following agent flow."""

from __future__ import annotations

import logging

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)

IFEVAL_SYSTEM_PROMPT = """\
Follow the instructions in the user's message exactly as specified.
Pay close attention to all formatting, content, and structural requirements."""


class IFEvalAgentFlow:
    """Instruction-following agent flow.

    Passes the prompt directly to the model with a minimal system prompt.
    Used for IFEval and IFBench benchmarks where the prompt itself contains
    the constraints to be verified.
    """

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        # IFEval uses "prompt", which may be mapped to "question" by field_map
        question = task.get("question", task.get("prompt", ""))

        response_text = ""
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": IFEVAL_SYSTEM_PROMPT},
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
ifeval_agent = IFEvalAgentFlow()
