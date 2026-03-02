"""Built-in VLM (Vision Language Model) agent flows.

These agents construct multimodal OpenAI API messages with image content
blocks alongside text, enabling evaluation of vision-language benchmarks.
"""

from __future__ import annotations

import base64
import logging
import os

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)

# Default root for rllm datasets (overridden by RLLM_HOME env var)
_DATASETS_ROOT = os.path.join(
    os.environ.get("RLLM_HOME", os.path.expanduser("~/.rllm")), "datasets"
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _load_image_as_data_uri(rel_path: str) -> str:
    """Load an image file and return a base64 data URI.

    Args:
        rel_path: Path relative to the datasets root directory.

    Returns:
        A ``data:image/png;base64,...`` URI string.
    """
    abs_path = os.path.join(_DATASETS_ROOT, rel_path)
    with open(abs_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _build_vlm_content(text: str, image_paths: list[str]) -> list[dict]:
    """Build OpenAI multimodal content blocks (image_url + text).

    Args:
        text: The text portion of the user message.
        image_paths: List of image paths relative to the datasets root.

    Returns:
        A list of content block dicts for the OpenAI API.
    """
    content: list[dict] = []

    for path in image_paths:
        try:
            data_uri = _load_image_as_data_uri(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": data_uri},
            })
        except Exception as e:
            logger.warning("Failed to load image %s: %s", path, e)

    content.append({"type": "text", "text": text})
    return content


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

VLM_MCQ_SYSTEM_PROMPT = """\
You are an expert problem solver with vision capabilities. You are given a multiple-choice \
question with one or more images and several answer options.
Analyze the images and question carefully, reason through the options, and select the best answer.
Respond with ONLY the letter of your chosen answer (e.g., A, B, C, D).
Do not include any other text in your final answer."""

VLM_MATH_SYSTEM_PROMPT = """\
You are a math problem solver with vision capabilities. You are given a math problem \
that may include one or more images.
Solve the problem step by step, showing your reasoning clearly.
Put your final answer in \\boxed{} notation.

For example: The answer is \\boxed{42}."""

VLM_OPEN_SYSTEM_PROMPT = """\
You are a helpful assistant with vision capabilities. You are given a question \
that may include one or more images.
Analyze the images carefully and answer the question accurately and concisely."""


# ---------------------------------------------------------------------------
# Agent flows
# ---------------------------------------------------------------------------


class VLMMCQAgentFlow:
    """VLM multiple-choice question answering agent flow.

    Handles VLM MCQ benchmarks (MMMU, MMMU-Pro). Expects task to have
    'question', 'images' (list of paths), and 'choices' fields.
    """

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        question = task.get("question", "")
        choices = task.get("choices", [])
        image_paths = task.get("images", [])

        # Format choices as A) ... B) ... C) ...
        formatted_choices = []
        for i, choice in enumerate(choices):
            letter = chr(ord("A") + i)
            formatted_choices.append(f"{letter}) {choice}")
        choices_text = "\n".join(formatted_choices)

        user_text = f"{question}\n\n{choices_text}" if choices_text else question

        # Build multimodal content
        if image_paths:
            user_content = _build_vlm_content(user_text, image_paths)
        else:
            user_content = user_text

        response_text = ""
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": VLM_MCQ_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("LLM call failed: %s", e)

        step = Step(input=user_text, output=response_text, done=True)
        traj = Trajectory(name="solver", steps=[step])
        return Episode(task=task, trajectories=[traj], artifacts={"answer": response_text})


class VLMMathAgentFlow:
    """VLM math reasoning agent flow.

    Handles VLM math benchmarks (MathVision, MathVista, DynaMath). Expects
    task to have 'question' and 'images' (list of paths).
    """

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        question = task.get("question", "")
        image_paths = task.get("images", [])

        # Build multimodal content
        if image_paths:
            user_content = _build_vlm_content(question, image_paths)
        else:
            user_content = question

        response_text = ""
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": VLM_MATH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("LLM call failed: %s", e)

        step = Step(input=question, output=response_text, done=True)
        traj = Trajectory(name="solver", steps=[step])
        return Episode(task=task, trajectories=[traj], artifacts={"answer": response_text})


class VLMOpenAgentFlow:
    """VLM open-ended question answering agent flow.

    Handles open-ended VLM benchmarks (ZEROBench, VLMs Are Blind, BabyVision).
    Expects task to have 'question' and 'images' (list of paths).
    """

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        question = task.get("question", "")
        image_paths = task.get("images", [])

        # Build multimodal content
        if image_paths:
            user_content = _build_vlm_content(question, image_paths)
        else:
            user_content = question

        response_text = ""
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": VLM_OPEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("LLM call failed: %s", e)

        step = Step(input=question, output=response_text, done=True)
        traj = Trajectory(name="solver", steps=[step])
        return Episode(task=task, trajectories=[traj], artifacts={"answer": response_text})


# Singleton instances for registry
vlm_mcq_agent = VLMMCQAgentFlow()
vlm_math_agent = VLMMathAgentFlow()
vlm_open_agent = VLMOpenAgentFlow()
