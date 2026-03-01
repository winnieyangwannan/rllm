"""Tests for IFEval agent flow and evaluator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rllm.experimental.eval.ifeval_evaluator import (
    IFEvalEvaluator,
    verify_instruction,
)
from rllm.experimental.eval.types import AgentConfig, AgentFlow, Evaluator
from rllm.types import Episode


def _mock_openai_response(content: str):
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
# IFEvalAgentFlow
# ---------------------------------------------------------------------------


class TestIFEvalAgentFlow:
    def test_returns_episode(self, base_config):
        from rllm.experimental.agents.ifeval_agent import ifeval_agent

        task = {"question": "Write a poem about cats. Use at least 100 words."}

        with patch("rllm.experimental.agents.ifeval_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("Cats are wonderful creatures...")
            MockOpenAI.return_value = mock_client

            result = ifeval_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert len(result.trajectories) == 1
        assert result.artifacts["answer"] == "Cats are wonderful creatures..."

    def test_is_agent_flow(self):
        from rllm.experimental.agents.ifeval_agent import ifeval_agent
        assert isinstance(ifeval_agent, AgentFlow)

    def test_uses_prompt_field(self, base_config):
        """Falls back to 'prompt' field if 'question' is not present."""
        from rllm.experimental.agents.ifeval_agent import ifeval_agent

        task = {"prompt": "Write something"}

        with patch("rllm.experimental.agents.ifeval_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("Something")
            MockOpenAI.return_value = mock_client

            result = ifeval_agent.run(task, base_config)

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        assert user_msg == "Write something"


# ---------------------------------------------------------------------------
# Instruction verification functions
# ---------------------------------------------------------------------------


class TestVerifyInstruction:
    def test_keywords_existence_pass(self):
        assert verify_instruction(
            "keywords:existence",
            "Hello world, this is a test",
            {"keywords": ["hello", "world"]},
        ) is True

    def test_keywords_existence_fail(self):
        assert verify_instruction(
            "keywords:existence",
            "Hello world",
            {"keywords": ["hello", "missing"]},
        ) is False

    def test_keywords_forbidden_pass(self):
        assert verify_instruction(
            "keywords:forbidden_words",
            "Hello world",
            {"forbidden_words": ["bad", "evil"]},
        ) is True

    def test_keywords_forbidden_fail(self):
        assert verify_instruction(
            "keywords:forbidden_words",
            "Hello bad world",
            {"forbidden_words": ["bad"]},
        ) is False

    def test_length_number_words_at_least(self):
        response = " ".join(["word"] * 50)
        assert verify_instruction(
            "length_constraints:number_words",
            response,
            {"num_words": 50, "relation": "at least"},
        ) is True

    def test_length_number_words_at_most_fail(self):
        response = " ".join(["word"] * 50)
        assert verify_instruction(
            "length_constraints:number_words",
            response,
            {"num_words": 10, "relation": "at most"},
        ) is False

    def test_change_case_lowercase(self):
        assert verify_instruction(
            "change_case:english_lowercase",
            "hello world 123",
            {},
        ) is True

    def test_change_case_lowercase_fail(self):
        assert verify_instruction(
            "change_case:english_lowercase",
            "Hello World",
            {},
        ) is False

    def test_change_case_uppercase(self):
        assert verify_instruction(
            "change_case:english_uppercase",
            "HELLO WORLD 123",
            {},
        ) is True

    def test_json_format(self):
        assert verify_instruction(
            "detectable_format:json_format",
            '{"key": "value"}',
            {},
        ) is True

    def test_json_format_in_code_block(self):
        assert verify_instruction(
            "detectable_format:json_format",
            'Here is the JSON:\n```json\n{"key": "value"}\n```',
            {},
        ) is True

    def test_no_comma(self):
        assert verify_instruction(
            "punctuation:no_comma",
            "Hello world. This is great.",
            {},
        ) is True

    def test_no_comma_fail(self):
        assert verify_instruction(
            "punctuation:no_comma",
            "Hello, world",
            {},
        ) is False

    def test_title(self):
        assert verify_instruction(
            "detectable_format:title",
            "<<My Title>>\nContent here",
            {},
        ) is True

    def test_postscript(self):
        assert verify_instruction(
            "detectable_content:postscript",
            "Main content\n\nP.S. Don't forget!",
            {},
        ) is True

    def test_unknown_instruction(self):
        # Unknown instructions pass (lenient mode)
        assert verify_instruction("unknown:type", "response", {}) is True

    def test_end_checker(self):
        assert verify_instruction(
            "startend:end_checker",
            "Some content ending with goodbye",
            {"end_phrase": "goodbye"},
        ) is True

    def test_keywords_frequency(self):
        assert verify_instruction(
            "keywords:frequency",
            "cat cat cat dog",
            {"keyword": "cat", "frequency": 3, "relation": "at least"},
        ) is True


# ---------------------------------------------------------------------------
# IFEvalEvaluator
# ---------------------------------------------------------------------------


class TestIFEvalEvaluator:
    def test_all_pass(self):
        evaluator = IFEvalEvaluator()
        task = {
            "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
            "kwargs": [{"keywords": ["hello"]}, {}],
        }
        ep = Episode(artifacts={"answer": "hello world"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True
        assert result.reward == 1.0
        assert any(s.name == "strict_accuracy" and s.value == 1.0 for s in result.signals)

    def test_partial_pass(self):
        evaluator = IFEvalEvaluator()
        task = {
            "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
            "kwargs": [{"keywords": ["hello"]}, {}],
        }
        ep = Episode(artifacts={"answer": "hello, world"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False
        assert result.reward == 0.0
        # Loose accuracy should be 0.5 (1 of 2 passed)
        loose = next(s for s in result.signals if s.name == "loose_accuracy")
        assert loose.value == pytest.approx(0.5)

    def test_no_instructions(self):
        evaluator = IFEvalEvaluator()
        task = {"instruction_id_list": [], "kwargs": []}
        ep = Episode(artifacts={"answer": "anything"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_is_evaluator(self):
        evaluator = IFEvalEvaluator()
        assert isinstance(evaluator, Evaluator)

    def test_metadata_has_results(self):
        evaluator = IFEvalEvaluator()
        task = {
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["test"]}],
        }
        ep = Episode(artifacts={"answer": "test response"})
        result = evaluator.evaluate(task, ep)
        assert "instruction_results" in result.metadata
        assert result.metadata["instruction_results"]["keywords:existence"] is True
