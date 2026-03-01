"""Built-in multiple-choice question agent flow."""

from __future__ import annotations

import logging

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)

MCQ_SYSTEM_PROMPT = """\
You are an expert problem solver. You are given a multiple-choice question with several answer options.
Analyze the question carefully, reason through the options, and select the best answer.
Respond with ONLY the letter of your chosen answer (e.g., A, B, C, D).
Do not include any other text in your final answer."""


class MCQAgentFlow:
    """Multiple-choice question answering agent flow.

    Handles any MCQ benchmark (MMLU-Pro, MMLU-Redux, GPQA, SuperGPQA, C-Eval, MMMLU).
    Expects task to have "question" and "choices" fields.
    """

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        question = task.get("question", "")
        choices = task.get("choices", [])

        # Format choices as A) ... B) ... C) ...
        formatted_choices = []
        for i, choice in enumerate(choices):
            letter = chr(ord("A") + i)
            formatted_choices.append(f"{letter}) {choice}")
        choices_text = "\n".join(formatted_choices)

        user_message = f"{question}\n\n{choices_text}" if choices_text else question

        response_text = ""
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": MCQ_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("LLM call failed: %s", e)

        step = Step(input=user_message, output=response_text, done=True)
        traj = Trajectory(name="solver", steps=[step])
        return Episode(task=task, trajectories=[traj], artifacts={"answer": response_text})


# Singleton instance for registry
mcq_agent = MCQAgentFlow()
