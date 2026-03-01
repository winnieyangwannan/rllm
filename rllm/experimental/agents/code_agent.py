"""Built-in code agent flow."""

from __future__ import annotations

import logging

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)

CODE_SYSTEM_PROMPT = """\
You are a competitive programming expert. Solve the given problem by writing a Python solution.
Read the problem carefully, think step by step, then provide your solution in a Python code block.

Your solution should read from stdin and write to stdout.
Wrap your code in ```python ... ``` markers."""


class CodeAgentFlow:
    """Single-turn code generation agent flow."""

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        question = task.get("question", "")

        response_text = ""
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": CODE_SYSTEM_PROMPT},
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
code_agent = CodeAgentFlow()
