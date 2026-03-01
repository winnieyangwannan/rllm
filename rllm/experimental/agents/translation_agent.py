"""Built-in translation agent flow."""

from __future__ import annotations

import logging

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)

TRANSLATION_SYSTEM_PROMPT = """\
You are an expert translator. Translate the given text accurately and naturally \
into the target language. Preserve the meaning, tone, and style of the original. \
Output only the translation, nothing else."""


class TranslationAgentFlow:
    """Translation agent flow for machine translation benchmarks."""

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        source_text = task.get("question", "")

        response_text = ""
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                    {"role": "user", "content": source_text},
                ],
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("LLM call failed: %s", e)

        step = Step(input=source_text, output=response_text, done=True)
        traj = Trajectory(name="translator", steps=[step])
        return Episode(task=task, trajectories=[traj], artifacts={"answer": response_text})


# Singleton instance for registry
translation_agent = TranslationAgentFlow()
