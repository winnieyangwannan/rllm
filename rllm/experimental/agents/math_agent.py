"""Built-in math agent flows."""

from __future__ import annotations

import logging

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)

MATH_SYSTEM_PROMPT = """\
You are a math problem solver. Solve the given problem step by step, showing your reasoning clearly.
Put your final answer in \\boxed{} notation.

For example: The answer is \\boxed{42}."""

COUNTDOWN_SYSTEM_PROMPT = """\
You are a math solver. You are given a target number and a set of numbers.
You must use each number exactly once and basic arithmetic operations (+, -, *, /) to reach the target.
Show your reasoning step by step, then provide your final equation inside <answer>...</answer> tags.

For example: <answer> (25 + 3) * 2 </answer>"""


class MathAgentFlow:
    """Single-turn math reasoning agent flow."""

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        question = task.get("question", "")

        response_text = ""
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": MATH_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("LLM call failed: %s", e)

        step = Step(input=question, output=response_text, done=True)
        traj = Trajectory(name="solver", steps=[step])
        return Episode(task=task, trajectories=[traj], artifacts={"answer": response_text})


class CountdownAgentFlow:
    """Countdown arithmetic agent flow."""

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        target = task.get("target", "")
        nums = task.get("nums", [])
        question = f"Target: {target}\nNumbers: {nums}"

        response_text = ""
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": COUNTDOWN_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("LLM call failed: %s", e)

        step = Step(input=question, output=response_text, done=True)
        traj = Trajectory(name="solver", steps=[step])
        return Episode(task=task, trajectories=[traj], artifacts={"answer": response_text})


# Singleton instances for registry
math_agent = MathAgentFlow()
countdown_agent = CountdownAgentFlow()
