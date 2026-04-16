"""Unit tests for RolloutClient text mode and OutputWithVersion overrides.

Tests added for Phase 3 (dual mode RolloutClient):
- Mode auto-detection based on tokenizer
- Model required validation in text mode
- OutputWithVersion override fields behavior
- LLMCallError handling from RolloutClient
"""

from __future__ import annotations

import asyncio
import sys

import pytest

sys.path.insert(0, "/home/winnieyangwn/rllm/agenthub/mle_agent")
sys.path.insert(0, "/home/winnieyangwn/rllm/examples/mlebench")


class TestRolloutClientModeDetection:
    """Test auto-detection of text/token mode based on tokenizer."""

    def test_text_mode_when_no_tokenizer(self):
        """Mode is 'text' when tokenizer is None."""
        from rllm.experimental.fully_async.client import RolloutClient

        client = RolloutClient(
            router_url="http://localhost:30000",
            model="test-model",
        )
        assert client.mode == "text"
        assert client._mode == "text"

    def test_token_mode_when_tokenizer_provided(self):
        """Mode is 'token' when tokenizer is provided."""
        from unittest.mock import MagicMock

        from rllm.experimental.fully_async.client import RolloutClient

        # Mock tokenizer with proper name_or_path for ToolParser
        mock_tokenizer = MagicMock()
        mock_tokenizer.name_or_path = "Qwen/Qwen3-8B"  # Needs to match a known model pattern

        client = RolloutClient(
            router_url="http://localhost:30000",
            tokenizer=mock_tokenizer,
        )
        assert client.mode == "token"
        assert client._mode == "token"


class TestRolloutClientValidation:
    """Test validation logic in RolloutClient."""

    def test_model_required_in_text_mode(self):
        """ValueError raised when model is None and tokenizer is None."""
        from rllm.experimental.fully_async.client import RolloutClient

        with pytest.raises(ValueError, match="model is required"):
            RolloutClient(router_url="http://localhost:30000")  # No model, no tokenizer

    def test_model_not_required_in_token_mode(self):
        """Model is optional when tokenizer is provided."""
        from unittest.mock import MagicMock

        from rllm.experimental.fully_async.client import RolloutClient

        # Mock tokenizer with proper name_or_path for ToolParser
        mock_tokenizer = MagicMock()
        mock_tokenizer.name_or_path = "Qwen/Qwen3-8B"

        # Should not raise
        client = RolloutClient(
            router_url="http://localhost:30000",
            tokenizer=mock_tokenizer,
        )
        assert client.model is None


class TestOutputWithVersionOverrides:
    """Test OutputWithVersion text mode override fields."""

    def test_override_completion_tokens(self):
        """get_completion_tokens() uses override when set."""
        from rllm.experimental.fully_async.protocol import OutputWithVersion

        output = OutputWithVersion(
            prompt_ids=[],
            output_chunks=[],
            _completion_token_count=150,
        )
        assert output.get_completion_tokens() == 150

    def test_override_input_context_size(self):
        """get_input_context_size() uses override when set."""
        from rllm.experimental.fully_async.protocol import OutputWithVersion

        output = OutputWithVersion(
            prompt_ids=[1, 2, 3],  # 3 tokens
            output_chunks=[],
            _input_context_size=500,
        )
        assert output.get_input_context_size() == 500  # Override, not len(prompt_ids)

    def test_to_sequence_returns_none_with_override(self):
        """to_sequence() returns None when override is set (text mode)."""
        from rllm.experimental.fully_async.protocol import OutputWithVersion

        output = OutputWithVersion(
            prompt_ids=[],
            output_chunks=[],
            _completion_token_count=100,
        )
        assert output.to_sequence() is None

    def test_to_sequence_returns_none_with_empty_chunks(self):
        """to_sequence() returns None when output_chunks is empty."""
        from rllm.experimental.fully_async.protocol import OutputWithVersion

        output = OutputWithVersion(
            prompt_ids=[1, 2, 3],
            output_chunks=[],
        )
        assert output.to_sequence() is None

    def test_no_override_uses_token_count(self):
        """Without override, get_completion_tokens() uses actual token count."""
        from rllm.experimental.fully_async.protocol import OutputChunk, OutputWithVersion

        output = OutputWithVersion(
            prompt_ids=[1, 2, 3],
            output_chunks=[
                OutputChunk(response_ids=[10, 11, 12], response_logprobs=[0.1, 0.2, 0.3], version=0),
            ],
        )
        assert output.get_completion_tokens() == 3  # len(response_ids)
        assert output.get_input_context_size() == 3  # len(prompt_ids)


class TestLLMCallErrorHandling:
    """Test that MLEBenchAgent catches LLMCallError from RolloutClient."""

    def test_rollout_llm_call_error_is_caught(self):
        """MLEBenchAgent catches LLMCallError from RolloutClient."""
        from mle_agent_loop import MLEBenchAgent

        from rllm.experimental.fully_async.client import LLMCallError as RolloutLLMCallError

        class FailingClient:
            async def chat_completion(self, messages, sampling_params=None, tools=None):
                raise RolloutLLMCallError("Server error", retryable=False)

        class DummySandbox:
            def exec(self, cmd, timeout=60):
                return "output"

            def close(self):
                pass

        agent = MLEBenchAgent(
            client=FailingClient(),
            sandbox=DummySandbox(),
            max_retries=0,
        )

        result = asyncio.run(agent.run([{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]))
        assert result.metrics["termination_reason"] == "model_call_error"

    def test_retryable_error_is_retried(self):
        """Retryable errors are retried before failing."""
        from mle_agent_loop import MLEBenchAgent

        from rllm.experimental.fully_async.client import LLMCallError as RolloutLLMCallError

        call_count = 0

        class RetryableFailingClient:
            async def chat_completion(self, messages, sampling_params=None, tools=None):
                nonlocal call_count
                call_count += 1
                raise RolloutLLMCallError("Rate limited", retryable=True)

        class DummySandbox:
            def exec(self, cmd, timeout=60):
                return "output"

            def close(self):
                pass

        agent = MLEBenchAgent(
            client=RetryableFailingClient(),
            sandbox=DummySandbox(),
            max_retries=2,
            retry_base_delay=0.01,  # Fast for tests
        )

        result = asyncio.run(agent.run([{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]))
        assert result.metrics["termination_reason"] == "model_call_error"
        assert call_count == 3  # 1 initial + 2 retries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
