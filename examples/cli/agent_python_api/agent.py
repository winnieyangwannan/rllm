"""Restaurant concierge agent implementing the AgentFlow protocol."""

from __future__ import annotations

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory


class ConciergeAgent:
    """A restaurant recommendation agent."""

    SYSTEM_PROMPT = "You are a helpful restaurant concierge. Given a user query about restaurants or food, provide a short, helpful recommendation. Always mention the cuisine type in your answer."

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")

        query = task.get("query", task.get("question", ""))
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        answer = response.choices[0].message.content or ""

        step = Step(input=query, output=answer, done=True)
        traj = Trajectory(name="concierge", steps=[step])
        return Episode(
            task=task,
            trajectories=[traj],
            artifacts={"answer": answer},
        )
