"""Tests for MCQ agent flow and MCQ evaluator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rllm.experimental.eval.types import (
    AgentConfig,
    AgentFlow,
    Evaluator,
    MCQEvaluator,
    Signal,
)
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


# ---------------------------------------------------------------------------
# MCQAgentFlow
# ---------------------------------------------------------------------------


class TestMCQAgentFlow:
    def test_returns_episode(self, base_config):
        from rllm.experimental.agents.mcq_agent import mcq_agent

        task = {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "ground_truth": "B",
            "data_source": "test",
        }

        with patch("rllm.experimental.agents.mcq_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("B")
            MockOpenAI.return_value = mock_client

            result = mcq_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert len(result.trajectories) == 1
        assert result.trajectories[0].steps[0].done is True
        assert result.artifacts["answer"] == "B"

    def test_formats_choices(self, base_config):
        from rllm.experimental.agents.mcq_agent import mcq_agent

        task = {
            "question": "Pick one",
            "choices": ["Alpha", "Beta", "Gamma"],
            "ground_truth": "A",
        }

        with patch("rllm.experimental.agents.mcq_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("A")
            MockOpenAI.return_value = mock_client

            mcq_agent.run(task, base_config)

            call_args = mock_client.chat.completions.create.call_args
            user_msg = call_args[1]["messages"][1]["content"]
            assert "A) Alpha" in user_msg
            assert "B) Beta" in user_msg
            assert "C) Gamma" in user_msg

    def test_llm_failure(self, base_config):
        from rllm.experimental.agents.mcq_agent import mcq_agent

        task = {"question": "Test", "choices": ["A", "B"], "ground_truth": "A"}

        with patch("rllm.experimental.agents.mcq_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("Connection error")
            MockOpenAI.return_value = mock_client

            result = mcq_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert result.artifacts["answer"] == ""

    def test_no_reward_computed(self, base_config):
        from rllm.experimental.agents.mcq_agent import mcq_agent

        task = {"question": "Test", "choices": ["A", "B"], "ground_truth": "A"}

        with patch("rllm.experimental.agents.mcq_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("A")
            MockOpenAI.return_value = mock_client

            result = mcq_agent.run(task, base_config)

        assert result.trajectories[0].reward is None

    def test_is_agent_flow(self):
        from rllm.experimental.agents.mcq_agent import mcq_agent
        assert isinstance(mcq_agent, AgentFlow)


# ---------------------------------------------------------------------------
# MCQEvaluator
# ---------------------------------------------------------------------------


class TestMCQEvaluator:
    def test_correct_answer(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "B", "data_source": "test"}
        ep = Episode(artifacts={"answer": "B"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True
        assert result.reward == 1.0

    def test_wrong_answer(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "B", "data_source": "test"}
        ep = Episode(artifacts={"answer": "A"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False
        assert result.reward == 0.0

    def test_answer_is_pattern(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "C"}
        ep = Episode(artifacts={"answer": "The answer is C"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_answer_is_colon_pattern(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "D"}
        ep = Episode(artifacts={"answer": "After analyzing, answer: D"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_bold_pattern(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "B"}
        ep = Episode(artifacts={"answer": "I think **B** is correct"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_parenthesized_pattern(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "A"}
        ep = Episode(artifacts={"answer": "The correct option is (A)"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_empty_answer(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "A"}
        ep = Episode(artifacts={"answer": ""})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False
        assert result.reward == 0.0

    def test_no_ground_truth(self):
        evaluator = MCQEvaluator()
        task = {}
        ep = Episode(artifacts={"answer": "A"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_lowercase_answer(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "B"}
        ep = Episode(artifacts={"answer": "b"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_ten_option_answer(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "J"}
        ep = Episode(artifacts={"answer": "J"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_signals_present(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "A"}
        ep = Episode(artifacts={"answer": "A"})
        result = evaluator.evaluate(task, ep)
        assert len(result.signals) > 0
        assert result.signals[0].name == "accuracy"

    def test_metadata_contains_answers(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "B"}
        ep = Episode(artifacts={"answer": "A"})
        result = evaluator.evaluate(task, ep)
        assert result.metadata["model_answer"] == "A"
        assert result.metadata["expected"] == "B"

    def test_is_evaluator(self):
        evaluator = MCQEvaluator()
        assert isinstance(evaluator, Evaluator)


# ---------------------------------------------------------------------------
# MCQEvaluator._extract_choice_letter edge cases
# ---------------------------------------------------------------------------


class TestExtractChoiceLetter:
    def test_single_letter(self):
        assert MCQEvaluator._extract_choice_letter("A") == "A"

    def test_single_letter_lowercase(self):
        assert MCQEvaluator._extract_choice_letter("c") == "C"

    def test_verbose_response(self):
        assert MCQEvaluator._extract_choice_letter("After careful analysis, the answer is B") == "B"

    def test_no_letter(self):
        assert MCQEvaluator._extract_choice_letter("No valid answer here") == ""

    def test_empty(self):
        assert MCQEvaluator._extract_choice_letter("") == ""

    def test_letter_in_word(self):
        # "A" as a standalone word (article) — should match as fallback
        result = MCQEvaluator._extract_choice_letter("A long explanation about biology")
        assert result == "A"

    def test_answer_with_parentheses(self):
        assert MCQEvaluator._extract_choice_letter("The answer is (D)") == "D"
