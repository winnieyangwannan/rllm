"""Tests for agentic benchmarks: BFCL agent/evaluator and multi-turn agent/LLM judge."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rllm.experimental.eval.bfcl_evaluator import BFCLEvaluator, _compare_function_calls
from rllm.experimental.eval.llm_judge_evaluator import LLMJudgeEvaluator
from rllm.experimental.eval.types import AgentConfig, AgentFlow, Evaluator
from rllm.types import Episode


def _mock_openai_response(content: str, tool_calls=None):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_choice.message.tool_calls = tool_calls
    mock_response.choices = [mock_choice]
    return mock_response


def _mock_tool_call(name: str, arguments: str):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


@pytest.fixture
def base_config():
    return AgentConfig(
        base_url="http://localhost:8000/v1",
        model="test-model",
        session_uid="test-001",
    )


# ---------------------------------------------------------------------------
# BFCLAgentFlow
# ---------------------------------------------------------------------------


class TestBFCLAgentFlow:
    def test_returns_episode(self, base_config):
        from rllm.experimental.agents.bfcl_agent import bfcl_agent

        task = {
            "question": [{"role": "user", "content": "Get the weather in Paris"}],
            "function": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
        }

        tool_call = _mock_tool_call("get_weather", '{"city": "Paris"}')
        with patch("rllm.experimental.agents.bfcl_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response(
                "", tool_calls=[tool_call]
            )
            MockOpenAI.return_value = mock_client

            result = bfcl_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert len(result.artifacts["tool_calls"]) == 1
        assert result.artifacts["tool_calls"][0]["name"] == "get_weather"

    def test_is_agent_flow(self):
        from rllm.experimental.agents.bfcl_agent import bfcl_agent
        assert isinstance(bfcl_agent, AgentFlow)

    def test_handles_no_tool_calls(self, base_config):
        from rllm.experimental.agents.bfcl_agent import bfcl_agent

        task = {"question": "Hello", "function": []}

        with patch("rllm.experimental.agents.bfcl_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("Hello!", tool_calls=None)
            MockOpenAI.return_value = mock_client

            result = bfcl_agent.run(task, base_config)

        assert result.artifacts["tool_calls"] == []
        assert result.artifacts["answer"] == "Hello!"


# ---------------------------------------------------------------------------
# BFCLEvaluator
# ---------------------------------------------------------------------------


class TestBFCLEvaluator:
    def test_correct_function_call(self):
        evaluator = BFCLEvaluator()
        task = {
            "ground_truth": [json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})],
        }
        ep = Episode(artifacts={
            "answer": "",
            "tool_calls": [{"name": "get_weather", "arguments": '{"city": "Paris"}'}],
        })
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True
        assert result.reward == 1.0

    def test_wrong_function_name(self):
        evaluator = BFCLEvaluator()
        task = {
            "ground_truth": [json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})],
        }
        ep = Episode(artifacts={
            "answer": "",
            "tool_calls": [{"name": "get_time", "arguments": '{"city": "Paris"}'}],
        })
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_no_model_calls(self):
        evaluator = BFCLEvaluator()
        task = {
            "ground_truth": [json.dumps({"name": "func", "arguments": {}})],
        }
        ep = Episode(artifacts={"answer": "", "tool_calls": []})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_no_ground_truth(self):
        evaluator = BFCLEvaluator()
        task = {"ground_truth": []}
        ep = Episode(artifacts={"answer": "", "tool_calls": [{"name": "f", "arguments": "{}"}]})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_is_evaluator(self):
        evaluator = BFCLEvaluator()
        assert isinstance(evaluator, Evaluator)

    def test_signals_present(self):
        evaluator = BFCLEvaluator()
        task = {"ground_truth": []}
        ep = Episode(artifacts={"answer": "", "tool_calls": []})
        result = evaluator.evaluate(task, ep)
        assert any(s.name == "ast_accuracy" for s in result.signals)


# ---------------------------------------------------------------------------
# _compare_function_calls
# ---------------------------------------------------------------------------


class TestCompareFunctionCalls:
    def test_exact_match(self):
        model = [{"name": "f", "arguments": {"a": 1}}]
        gt = [json.dumps({"name": "f", "arguments": {"a": 1}})]
        is_correct, _ = _compare_function_calls(model, gt)
        assert is_correct is True

    def test_no_match(self):
        model = [{"name": "f", "arguments": {"a": 1}}]
        gt = [json.dumps({"name": "g", "arguments": {"a": 1}})]
        is_correct, _ = _compare_function_calls(model, gt)
        assert is_correct is False

    def test_empty_model(self):
        gt = [json.dumps({"name": "f", "arguments": {}})]
        is_correct, _ = _compare_function_calls([], gt)
        assert is_correct is False

    def test_empty_gt(self):
        is_correct, _ = _compare_function_calls([{"name": "f"}], [])
        assert is_correct is True


# ---------------------------------------------------------------------------
# MultiturnAgentFlow
# ---------------------------------------------------------------------------


class TestMultiturnAgentFlow:
    def test_returns_episode(self, base_config):
        from rllm.experimental.agents.multiturn_agent import multiturn_agent

        task = {
            "turns": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
            ],
        }

        with patch("rllm.experimental.agents.multiturn_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                _mock_openai_response("Hi there!"),
                _mock_openai_response("I'm doing well, thanks!"),
            ]
            MockOpenAI.return_value = mock_client

            result = multiturn_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert len(result.trajectories[0].steps) == 2
        assert result.artifacts["answer"] == "I'm doing well, thanks!"

    def test_is_agent_flow(self):
        from rllm.experimental.agents.multiturn_agent import multiturn_agent
        assert isinstance(multiturn_agent, AgentFlow)

    def test_single_turn_fallback(self, base_config):
        from rllm.experimental.agents.multiturn_agent import multiturn_agent

        task = {"question": "Hello"}

        with patch("rllm.experimental.agents.multiturn_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("Hi!")
            MockOpenAI.return_value = mock_client

            result = multiturn_agent.run(task, base_config)

        assert result.artifacts["answer"] == "Hi!"


# ---------------------------------------------------------------------------
# LLMJudgeEvaluator
# ---------------------------------------------------------------------------


class TestLLMJudgeEvaluator:
    def test_no_rubric_fallback(self):
        evaluator = LLMJudgeEvaluator()
        task = {"question": "Hello"}
        ep = Episode(artifacts={"answer": "Hi there!"})
        result = evaluator.evaluate(task, ep)
        # No rubric → passes if response is non-empty
        assert result.is_correct is True
        assert result.metadata.get("reason") == "no_rubric_available"

    def test_empty_answer_no_rubric(self):
        evaluator = LLMJudgeEvaluator()
        task = {"question": "Hello"}
        ep = Episode(artifacts={"answer": ""})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_is_evaluator(self):
        evaluator = LLMJudgeEvaluator()
        assert isinstance(evaluator, Evaluator)

    def test_signals_present(self):
        evaluator = LLMJudgeEvaluator()
        task = {"question": "Hello"}
        ep = Episode(artifacts={"answer": "Hi"})
        result = evaluator.evaluate(task, ep)
        assert any(s.name == "judge_score" for s in result.signals)

    def test_with_rubric_no_judge(self):
        evaluator = LLMJudgeEvaluator()  # No judge_base_url
        task = {"question": "Hello", "rubric": "Should greet politely"}
        ep = Episode(artifacts={"answer": "Hi there!"})
        result = evaluator.evaluate(task, ep)
        # Falls back since no judge available
        assert result.metadata.get("reason") == "judge_unavailable_fallback"
