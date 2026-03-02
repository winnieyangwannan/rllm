"""Unit tests for the concierge agent — no rLLM infra needed."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from concierge_agent.agent import ConciergeAgent
from concierge_agent.evaluator import RelevanceEvaluator
from rllm.experimental.eval.types import AgentConfig


def _make_config() -> AgentConfig:
    return AgentConfig(base_url="http://fake:8000/v1", model="test-model", session_uid="test-session")


def _mock_completion(content: str) -> MagicMock:
    """Build a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestConciergeAgent:
    @patch("concierge_agent.agent.OpenAI")
    def test_returns_episode(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_completion(
            "I recommend Sakura — a great Japanese sushi restaurant."
        )
        mock_openai_cls.return_value = mock_client

        agent = ConciergeAgent()
        episode = agent.run({"query": "Best sushi?"}, _make_config())

        assert len(episode.trajectories) == 1
        assert episode.artifacts["answer"] == "I recommend Sakura — a great Japanese sushi restaurant."
        assert episode.trajectories[0].name == "concierge"

    @patch("concierge_agent.agent.OpenAI")
    def test_passes_model_and_base_url(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_completion("Try La Piazza!")
        mock_openai_cls.return_value = mock_client

        agent = ConciergeAgent()
        config = _make_config()
        agent.run({"query": "Italian food?"}, config)

        mock_openai_cls.assert_called_once_with(base_url="http://fake:8000/v1", api_key="EMPTY")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "test-model"


class TestRelevanceEvaluator:
    @patch("concierge_agent.agent.OpenAI")
    def test_correct_cuisine(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_completion(
            "Try Sakura for authentic Japanese cuisine."
        )
        mock_openai_cls.return_value = mock_client

        agent = ConciergeAgent()
        task = {"query": "Best sushi?", "cuisine": "japanese"}
        episode = agent.run(task, _make_config())

        evaluator = RelevanceEvaluator()
        result = evaluator.evaluate(task, episode)

        assert result.reward == 1.0
        assert result.is_correct is True

    @patch("concierge_agent.agent.OpenAI")
    def test_wrong_cuisine(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_completion(
            "Try the burger at Joe's Diner."
        )
        mock_openai_cls.return_value = mock_client

        agent = ConciergeAgent()
        task = {"query": "Best sushi?", "cuisine": "japanese"}
        episode = agent.run(task, _make_config())

        evaluator = RelevanceEvaluator()
        result = evaluator.evaluate(task, episode)

        assert result.reward == 0.0
        assert result.is_correct is False
