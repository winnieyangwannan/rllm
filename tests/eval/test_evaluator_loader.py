"""Tests for evaluator loader: registry, import paths, and catalog resolution."""

from __future__ import annotations

import pytest

from rllm.experimental.eval.evaluator_loader import (
    _EVALUATOR_REGISTRY,
    load_evaluator,
    resolve_evaluator_from_catalog,
)
from rllm.experimental.eval.types import (
    CodeEvaluator,
    CountdownEvaluator,
    Evaluator,
    F1Evaluator,
    MathEvaluator,
)


class TestLoadEvaluator:
    def test_load_math_evaluator_by_name(self):
        evaluator = load_evaluator("math_reward_fn")
        assert isinstance(evaluator, MathEvaluator)

    def test_load_countdown_evaluator_by_name(self):
        evaluator = load_evaluator("countdown_reward_fn")
        assert isinstance(evaluator, CountdownEvaluator)

    def test_load_code_evaluator_by_name(self):
        evaluator = load_evaluator("code_reward_fn")
        assert isinstance(evaluator, CodeEvaluator)

    def test_load_f1_evaluator_by_name(self):
        evaluator = load_evaluator("f1_reward_fn")
        assert isinstance(evaluator, F1Evaluator)

    def test_load_by_import_path(self):
        evaluator = load_evaluator("rllm.experimental.eval.types:MathEvaluator")
        assert isinstance(evaluator, MathEvaluator)

    def test_load_unknown_name_raises(self):
        with pytest.raises(KeyError, match="not found in registry"):
            load_evaluator("nonexistent_evaluator")

    def test_load_bad_import_path_raises(self):
        with pytest.raises(ImportError):
            load_evaluator("nonexistent.module:MyEvaluator")

    def test_load_object_without_evaluate_raises(self):
        # EvalOutput is a dataclass with required args, so instantiation fails with TypeError
        with pytest.raises(TypeError):
            load_evaluator("rllm.experimental.eval.types:EvalOutput")

    def test_all_registry_entries_are_evaluators(self):
        for name in _EVALUATOR_REGISTRY:
            evaluator = load_evaluator(name)
            assert isinstance(evaluator, Evaluator), f"{name} is not an Evaluator"


class TestResolveEvaluatorFromCatalog:
    def test_resolve_gsm8k(self):
        evaluator = resolve_evaluator_from_catalog("gsm8k")
        assert isinstance(evaluator, MathEvaluator)

    def test_resolve_countdown(self):
        evaluator = resolve_evaluator_from_catalog("countdown")
        assert isinstance(evaluator, CountdownEvaluator)

    def test_resolve_deepcoder(self):
        evaluator = resolve_evaluator_from_catalog("deepcoder")
        assert isinstance(evaluator, CodeEvaluator)

    def test_resolve_hotpotqa(self):
        evaluator = resolve_evaluator_from_catalog("hotpotqa")
        assert isinstance(evaluator, F1Evaluator)

    def test_resolve_unknown_returns_none(self):
        result = resolve_evaluator_from_catalog("unknown_benchmark")
        assert result is None
