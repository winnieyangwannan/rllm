"""Tests for built-in agent flows."""

from unittest.mock import MagicMock, patch

import pytest

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory


def _mock_openai_response(content: str):
    """Create a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def base_config():
    return AgentConfig(
        base_url="http://localhost:8000/v1",
        model="test-model",
        session_uid="test-001",
    )


class TestMathAgentFlow:
    def test_returns_episode(self, base_config):
        from rllm.experimental.agents.math_agent import math_agent

        task = {"question": "What is 2+2?", "ground_truth": "4", "data_source": "test"}

        with patch("rllm.experimental.agents.math_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("The answer is \\boxed{4}")
            MockOpenAI.return_value = mock_client

            result = math_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert len(result.trajectories) == 1
        assert result.trajectories[0].steps[0].done is True
        assert result.artifacts["answer"] == "The answer is \\boxed{4}"

    def test_llm_failure(self, base_config):
        from rllm.experimental.agents.math_agent import math_agent

        task = {"question": "What is 2+2?", "ground_truth": "4", "data_source": "test"}

        with patch("rllm.experimental.agents.math_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("Connection error")
            MockOpenAI.return_value = mock_client

            result = math_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert result.artifacts["answer"] == ""

    def test_no_reward_computed(self, base_config):
        """AgentFlow should NOT compute reward — that's the evaluator's job."""
        from rllm.experimental.agents.math_agent import math_agent

        task = {"question": "What is 2+2?", "ground_truth": "4", "data_source": "test"}

        with patch("rllm.experimental.agents.math_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("The answer is \\boxed{4}")
            MockOpenAI.return_value = mock_client

            result = math_agent.run(task, base_config)

        # Reward should be None (not set by agent)
        assert result.trajectories[0].reward is None


class TestCountdownAgentFlow:
    def test_returns_episode(self, base_config):
        from rllm.experimental.agents.math_agent import countdown_agent

        task = {"target": 10, "nums": [2, 3, 5], "data_source": "countdown"}

        with patch("rllm.experimental.agents.math_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("Let me think... <answer>2 * 5</answer>")
            MockOpenAI.return_value = mock_client

            result = countdown_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert len(result.trajectories) == 1
        assert "answer" in result.artifacts


class TestQAAgentFlow:
    def test_returns_episode(self, base_config):
        from rllm.experimental.agents.qa_agent import qa_agent

        task = {"question": "Who wrote Hamlet?", "ground_truth": "Shakespeare", "data_source": "test"}

        with patch("rllm.experimental.agents.qa_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("Shakespeare wrote Hamlet.")
            MockOpenAI.return_value = mock_client

            result = qa_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert len(result.trajectories) == 1
        assert result.artifacts["answer"] == "Shakespeare wrote Hamlet."


class TestCodeAgentFlow:
    def test_returns_episode(self, base_config):
        from rllm.experimental.agents.code_agent import code_agent

        task = {
            "question": "Write a function that adds two numbers",
            "ground_truth": {"inputs": ["1 2"], "outputs": ["3"]},
            "data_source": "taco",
        }

        with patch("rllm.experimental.agents.code_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("```python\nprint(sum(map(int, input().split())))\n```")
            MockOpenAI.return_value = mock_client

            result = code_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert len(result.trajectories) == 1
        assert isinstance(result.trajectories[0], Trajectory)
