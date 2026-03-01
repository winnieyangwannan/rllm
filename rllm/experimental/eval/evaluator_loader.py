"""Evaluator loader: resolves evaluator by registry name or import path."""

from __future__ import annotations

import importlib
import json
import os

from rllm.experimental.eval.bfcl_evaluator import BFCLEvaluator
from rllm.experimental.eval.ifeval_evaluator import IFEvalEvaluator
from rllm.experimental.eval.llm_judge_evaluator import LLMJudgeEvaluator
from rllm.experimental.eval.types import (
    CodeEvaluator,
    CountdownEvaluator,
    Evaluator,
    F1Evaluator,
    MCQEvaluator,
    MathEvaluator,
)

_EVALUATOR_REGISTRY: dict[str, type] = {
    "math_reward_fn": MathEvaluator,
    "countdown_reward_fn": CountdownEvaluator,
    "code_reward_fn": CodeEvaluator,
    "f1_reward_fn": F1Evaluator,
    "mcq_reward_fn": MCQEvaluator,
    "ifeval_reward_fn": IFEvalEvaluator,
    "bfcl_reward_fn": BFCLEvaluator,
    "llm_judge_reward_fn": LLMJudgeEvaluator,
}


def _load_dataset_catalog() -> dict:
    """Load the datasets.json catalog from the registry directory."""
    catalog_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "registry",
        "datasets.json",
    )
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def load_evaluator(name_or_path: str) -> Evaluator:
    """Load an evaluator by registry name or import path.

    Args:
        name_or_path: Either a registry name (e.g., "math_reward_fn") or a
            colon-separated import path (e.g., "my_module:MyEvaluator").

    Returns:
        An Evaluator instance with an .evaluate() method.
    """
    if ":" in name_or_path:
        module_path, attr_name = name_or_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, attr_name)
        # If it's a class, instantiate it; if it's an instance, use as-is
        if isinstance(obj, type):
            obj = obj()
        if not hasattr(obj, "evaluate") or not callable(obj.evaluate):
            raise TypeError(
                f"Evaluator '{name_or_path}' must have an .evaluate() method, "
                f"got {type(obj).__name__}"
            )
        return obj

    if name_or_path in _EVALUATOR_REGISTRY:
        return _EVALUATOR_REGISTRY[name_or_path]()

    available = ", ".join(sorted(_EVALUATOR_REGISTRY.keys()))
    raise KeyError(f"Evaluator '{name_or_path}' not found in registry. Available: {available}")


def resolve_evaluator_from_catalog(benchmark: str) -> Evaluator | None:
    """Auto-resolve an evaluator from the datasets.json reward_fn field.

    Args:
        benchmark: Dataset name (e.g., "gsm8k").

    Returns:
        An Evaluator instance if the benchmark's reward_fn maps to a known evaluator,
        None otherwise.
    """
    try:
        catalog = _load_dataset_catalog()
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    entry = catalog.get("datasets", {}).get(benchmark)
    if entry is None:
        return None

    reward_fn_name = entry.get("reward_fn")
    if reward_fn_name is None:
        return None

    evaluator_cls = _EVALUATOR_REGISTRY.get(reward_fn_name)
    if evaluator_cls is None:
        return None

    return evaluator_cls()
