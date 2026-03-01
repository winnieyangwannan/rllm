"""Built-in multi-turn conversation agent flow."""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)

MULTITURN_SYSTEM_PROMPT = """\
You are a helpful assistant engaged in a multi-turn conversation. \
Respond thoughtfully and consistently across turns. \
Maintain coherence with your previous responses."""


class MultiturnAgentFlow:
    """Multi-turn conversation agent flow.

    Handles benchmarks that require multi-turn conversations (e.g., MultiChallenge).
    Reads conversation history from the task and sends messages turn by turn,
    collecting model responses.
    """

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")

        # Get conversation turns from task
        # MultiChallenge format: list of turns or conversation history
        turns = task.get("turns", [])
        question = task.get("question", "")

        # If no turns but has question, treat as single turn
        if not turns and question:
            turns = [{"role": "user", "content": question}]

        messages = [{"role": "system", "content": MULTITURN_SYSTEM_PROMPT}]
        steps = []
        final_response = ""

        try:
            for turn in turns:
                if isinstance(turn, dict):
                    messages.append(turn)
                elif isinstance(turn, str):
                    messages.append({"role": "user", "content": turn})

                # Only generate a response after user messages
                if messages[-1].get("role") == "user":
                    response = client.chat.completions.create(
                        model=config.model,
                        messages=messages,
                    )
                    response_text = response.choices[0].message.content or ""
                    messages.append({"role": "assistant", "content": response_text})
                    final_response = response_text

                    step = Step(
                        input=messages[-2].get("content", ""),
                        output=response_text,
                        done=False,
                    )
                    steps.append(step)
        except Exception as e:
            logger.warning("LLM call failed: %s", e)

        # Mark last step as done
        if steps:
            steps[-1] = Step(
                input=steps[-1].input,
                output=steps[-1].output,
                done=True,
            )

        traj = Trajectory(name="conversation", steps=steps)
        return Episode(
            task=task,
            trajectories=[traj],
            artifacts={
                "answer": final_response,
                "conversation": [
                    {"role": m["role"], "content": m.get("content", "")}
                    for m in messages
                ],
            },
        )


# Singleton instance for registry
multiturn_agent = MultiturnAgentFlow()
