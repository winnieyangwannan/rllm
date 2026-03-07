"""Restaurant concierge agent implementing the AgentFlow protocol.

This agent receives a restaurant query and returns a recommendation
using a standard OpenAI-compatible API call.
"""

from __future__ import annotations

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory


class ConciergeAgent:
    """A restaurant recommendation agent."""

    SYSTEM_PROMPT = "You are a helpful restaurant concierge. Given a user query about restaurants or food, provide a short, helpful recommendation. Always mention the cuisine type in your answer."

    def run(self, task: dict, config: AgentConfig) -> Episode:
        """Run the agent on a single task.

        Args:
            task: Dict with at least a "query" field.
            config: AgentConfig with base_url, model, session_uid.

        Returns:
            An Episode with one trajectory containing the agent's answer.
        """
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


# Module-level instance for entry-point discovery
concierge_agent = ConciergeAgent()
