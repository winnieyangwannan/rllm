"""Tests for VLM agent flows."""

from __future__ import annotations

import base64
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from rllm.experimental.eval.types import AgentConfig, AgentFlow
from rllm.types import Episode


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
        session_uid="test-vlm-001",
    )


@pytest.fixture
def image_dir(tmp_path, monkeypatch):
    """Create a temporary dataset directory with a test image."""
    datasets_dir = tmp_path / "datasets"
    img_dir = datasets_dir / "test_bench" / "images"
    img_dir.mkdir(parents=True)

    # Create a minimal 1x1 PNG file
    import struct
    import zlib

    def _make_png():
        sig = b"\x89PNG\r\n\x1a\n"
        # IHDR
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        # IDAT
        raw_data = zlib.compress(b"\x00\xff\x00\x00")
        idat_crc = zlib.crc32(b"IDAT" + raw_data) & 0xFFFFFFFF
        idat = struct.pack(">I", len(raw_data)) + b"IDAT" + raw_data + struct.pack(">I", idat_crc)
        # IEND
        iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
        return sig + ihdr + idat + iend

    png_bytes = _make_png()
    (img_dir / "test_0_image.png").write_bytes(png_bytes)
    (img_dir / "test_1_image.png").write_bytes(png_bytes)

    # Patch the _DATASETS_ROOT in vlm_agent module
    monkeypatch.setattr(
        "rllm.experimental.agents.vlm_agent._DATASETS_ROOT",
        str(datasets_dir),
    )

    return str(datasets_dir)


# ---------------------------------------------------------------------------
# VLMMCQAgentFlow
# ---------------------------------------------------------------------------


class TestVLMMCQAgentFlow:
    def test_returns_episode(self, base_config, image_dir):
        from rllm.experimental.agents.vlm_agent import vlm_mcq_agent

        task = {
            "question": "What shape is shown in the image?",
            "images": ["test_bench/images/test_0_image.png"],
            "choices": ["Circle", "Square", "Triangle", "Rectangle"],
            "ground_truth": "B",
            "data_source": "test",
        }

        with patch("rllm.experimental.agents.vlm_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("B")
            MockOpenAI.return_value = mock_client

            result = vlm_mcq_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert len(result.trajectories) == 1
        assert result.trajectories[0].steps[0].done is True
        assert result.artifacts["answer"] == "B"

    def test_multimodal_message_construction(self, base_config, image_dir):
        from rllm.experimental.agents.vlm_agent import vlm_mcq_agent

        task = {
            "question": "What is shown?",
            "images": ["test_bench/images/test_0_image.png"],
            "choices": ["A thing", "Another thing"],
            "ground_truth": "A",
        }

        with patch("rllm.experimental.agents.vlm_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("A")
            MockOpenAI.return_value = mock_client

            vlm_mcq_agent.run(task, base_config)

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            user_msg = messages[1]
            assert user_msg["role"] == "user"
            # Content should be a list with image_url and text blocks
            content = user_msg["content"]
            assert isinstance(content, list)
            assert content[0]["type"] == "image_url"
            assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
            assert content[1]["type"] == "text"
            assert "A) A thing" in content[1]["text"]
            assert "B) Another thing" in content[1]["text"]

    def test_formats_choices(self, base_config, image_dir):
        from rllm.experimental.agents.vlm_agent import vlm_mcq_agent

        task = {
            "question": "Pick one",
            "images": ["test_bench/images/test_0_image.png"],
            "choices": ["Alpha", "Beta", "Gamma"],
            "ground_truth": "A",
        }

        with patch("rllm.experimental.agents.vlm_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("A")
            MockOpenAI.return_value = mock_client

            vlm_mcq_agent.run(task, base_config)

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            content = messages[1]["content"]
            text_block = content[-1]["text"]
            assert "A) Alpha" in text_block
            assert "B) Beta" in text_block
            assert "C) Gamma" in text_block

    def test_no_images_falls_back_to_text(self, base_config):
        from rllm.experimental.agents.vlm_agent import vlm_mcq_agent

        task = {
            "question": "Plain text question",
            "images": [],
            "choices": ["Yes", "No"],
            "ground_truth": "A",
        }

        with patch("rllm.experimental.agents.vlm_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("A")
            MockOpenAI.return_value = mock_client

            result = vlm_mcq_agent.run(task, base_config)

            call_args = mock_client.chat.completions.create.call_args
            user_content = call_args[1]["messages"][1]["content"]
            # Without images, content should be a plain string
            assert isinstance(user_content, str)

        assert isinstance(result, Episode)
        assert result.artifacts["answer"] == "A"

    def test_llm_failure(self, base_config, image_dir):
        from rllm.experimental.agents.vlm_agent import vlm_mcq_agent

        task = {
            "question": "Test",
            "images": ["test_bench/images/test_0_image.png"],
            "choices": ["A", "B"],
            "ground_truth": "A",
        }

        with patch("rllm.experimental.agents.vlm_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("Connection error")
            MockOpenAI.return_value = mock_client

            result = vlm_mcq_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert result.artifacts["answer"] == ""

    def test_is_agent_flow(self):
        from rllm.experimental.agents.vlm_agent import vlm_mcq_agent
        assert isinstance(vlm_mcq_agent, AgentFlow)


# ---------------------------------------------------------------------------
# VLMMathAgentFlow
# ---------------------------------------------------------------------------


class TestVLMMathAgentFlow:
    def test_returns_episode(self, base_config, image_dir):
        from rllm.experimental.agents.vlm_agent import vlm_math_agent

        task = {
            "question": "What is the area of the shape in the image?",
            "images": ["test_bench/images/test_0_image.png"],
            "ground_truth": "42",
            "data_source": "test",
        }

        with patch("rllm.experimental.agents.vlm_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response(
                "The area is \\boxed{42}"
            )
            MockOpenAI.return_value = mock_client

            result = vlm_math_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert "\\boxed{42}" in result.artifacts["answer"]

    def test_multimodal_message_construction(self, base_config, image_dir):
        from rllm.experimental.agents.vlm_agent import vlm_math_agent

        task = {
            "question": "Solve the problem in the image",
            "images": [
                "test_bench/images/test_0_image.png",
                "test_bench/images/test_1_image.png",
            ],
            "ground_truth": "7",
        }

        with patch("rllm.experimental.agents.vlm_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("\\boxed{7}")
            MockOpenAI.return_value = mock_client

            vlm_math_agent.run(task, base_config)

            call_args = mock_client.chat.completions.create.call_args
            content = call_args[1]["messages"][1]["content"]
            assert isinstance(content, list)
            # Two images + one text block
            image_blocks = [b for b in content if b["type"] == "image_url"]
            text_blocks = [b for b in content if b["type"] == "text"]
            assert len(image_blocks) == 2
            assert len(text_blocks) == 1

    def test_no_reward_computed(self, base_config, image_dir):
        from rllm.experimental.agents.vlm_agent import vlm_math_agent

        task = {
            "question": "Test",
            "images": ["test_bench/images/test_0_image.png"],
            "ground_truth": "4",
        }

        with patch("rllm.experimental.agents.vlm_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("\\boxed{4}")
            MockOpenAI.return_value = mock_client

            result = vlm_math_agent.run(task, base_config)

        assert result.trajectories[0].reward is None

    def test_is_agent_flow(self):
        from rllm.experimental.agents.vlm_agent import vlm_math_agent
        assert isinstance(vlm_math_agent, AgentFlow)


# ---------------------------------------------------------------------------
# VLMOpenAgentFlow
# ---------------------------------------------------------------------------


class TestVLMOpenAgentFlow:
    def test_returns_episode(self, base_config, image_dir):
        from rllm.experimental.agents.vlm_agent import vlm_open_agent

        task = {
            "question": "Describe what you see in the image",
            "images": ["test_bench/images/test_0_image.png"],
            "ground_truth": "A red circle",
            "data_source": "test",
        }

        with patch("rllm.experimental.agents.vlm_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response(
                "I see a red circle"
            )
            MockOpenAI.return_value = mock_client

            result = vlm_open_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert result.artifacts["answer"] == "I see a red circle"

    def test_multimodal_message(self, base_config, image_dir):
        from rllm.experimental.agents.vlm_agent import vlm_open_agent

        task = {
            "question": "What number is shown?",
            "images": ["test_bench/images/test_0_image.png"],
            "ground_truth": "5",
        }

        with patch("rllm.experimental.agents.vlm_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("5")
            MockOpenAI.return_value = mock_client

            vlm_open_agent.run(task, base_config)

            call_args = mock_client.chat.completions.create.call_args
            content = call_args[1]["messages"][1]["content"]
            assert isinstance(content, list)
            assert content[0]["type"] == "image_url"
            assert content[1]["type"] == "text"
            assert content[1]["text"] == "What number is shown?"

    def test_llm_failure(self, base_config, image_dir):
        from rllm.experimental.agents.vlm_agent import vlm_open_agent

        task = {
            "question": "Test",
            "images": ["test_bench/images/test_0_image.png"],
            "ground_truth": "answer",
        }

        with patch("rllm.experimental.agents.vlm_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("Error")
            MockOpenAI.return_value = mock_client

            result = vlm_open_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert result.artifacts["answer"] == ""

    def test_is_agent_flow(self):
        from rllm.experimental.agents.vlm_agent import vlm_open_agent
        assert isinstance(vlm_open_agent, AgentFlow)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestBuildVLMContent:
    def test_single_image(self, image_dir):
        from rllm.experimental.agents.vlm_agent import _build_vlm_content

        content = _build_vlm_content("hello", ["test_bench/images/test_0_image.png"])
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "hello"

    def test_multiple_images(self, image_dir):
        from rllm.experimental.agents.vlm_agent import _build_vlm_content

        content = _build_vlm_content(
            "question",
            [
                "test_bench/images/test_0_image.png",
                "test_bench/images/test_1_image.png",
            ],
        )
        assert len(content) == 3
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "text"

    def test_no_images(self, image_dir):
        from rllm.experimental.agents.vlm_agent import _build_vlm_content

        content = _build_vlm_content("text only", [])
        assert len(content) == 1
        assert content[0]["type"] == "text"

    def test_missing_image_skipped(self, image_dir):
        from rllm.experimental.agents.vlm_agent import _build_vlm_content

        content = _build_vlm_content("text", ["nonexistent/path.png"])
        # Missing image is skipped, text remains
        assert len(content) == 1
        assert content[0]["type"] == "text"


class TestLoadImageAsDataURI:
    def test_loads_image(self, image_dir):
        from rllm.experimental.agents.vlm_agent import _load_image_as_data_uri

        uri = _load_image_as_data_uri("test_bench/images/test_0_image.png")
        assert uri.startswith("data:image/png;base64,")
        # Verify it's valid base64
        b64_part = uri.split(",", 1)[1]
        data = base64.b64decode(b64_part)
        assert data[:4] == b"\x89PNG"
