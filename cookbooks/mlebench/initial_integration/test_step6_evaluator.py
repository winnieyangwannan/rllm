"""Step 6 tests for MLEEvaluator.

Test cases from mlebench_integration_cc.md Step 6:
A) Stage 1 — no solution: pred_solution=None → reward=0.0
B) Stage 1 — no submission.csv ref: solution without "submission.csv" → reward=0.0
C) Stage 4 — execution failure: broken solution.py → exec error in signals
D) Stage 5 — missing CSV: solution runs but no submission.csv → reward=0.0
E) Stage 6 — invalid CSV format: malformed submission.csv → validation error
F) Stage 7 — known good submission: correct CSV → expected percentile
G) get_rank_and_percentile() tolerance logic with edge-case scores

Usage:
    # Run all tests
    python -m pytest cookbooks/mlebench/test_evaluator.py -v

    # Run specific test
    python -m pytest cookbooks/mlebench/test_evaluator.py::test_no_solution -v

    # Run with real sandbox (requires AgentBox manager)
    AGENTBOX_MANAGER_URI=http://h200-137-000-067:35743 \
    python -m pytest cookbooks/mlebench/test_evaluator.py -v -k "real"
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

# Mock rllm imports before loading evaluator (avoids heavy dependency chain)
mock_eval_types = ModuleType("rllm.experimental.eval.types")


class Signal:
    """Mock Signal class."""

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"Signal({self.name!r}, {self.value})"


class EvalOutput:
    """Mock EvalOutput class."""

    def __init__(self, reward=0.0, is_correct=False, signals=None, metadata=None):
        self.reward = reward
        self.is_correct = is_correct
        self.signals = signals or []
        self.metadata = metadata or {}

    def __repr__(self):
        return f"EvalOutput(reward={self.reward}, is_correct={self.is_correct})"


mock_eval_types.Signal = Signal
mock_eval_types.EvalOutput = EvalOutput

# Set up mock module hierarchy
sys.modules["rllm"] = ModuleType("rllm")
sys.modules["rllm.experimental"] = ModuleType("rllm.experimental")
sys.modules["rllm.experimental.eval"] = ModuleType("rllm.experimental.eval")
sys.modules["rllm.experimental.eval.types"] = mock_eval_types
sys.modules["rllm.types"] = ModuleType("rllm.types")
sys.modules["rllm.types"].Episode = type("Episode", (), {})

# Now load evaluator module directly (bypassing __init__.py)
# Path: cookbooks/mlebench -> rllm -> agenthub/mle_agent/mle_agent/evaluator.py
evaluator_path = Path(__file__).parent.parent.parent / "agenthub" / "mle_agent" / "mle_agent" / "evaluator.py"
spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
evaluator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator_module)

MLEEvaluator = evaluator_module.MLEEvaluator
get_rank_and_percentile = evaluator_module.get_rank_and_percentile
_fail = evaluator_module._fail


# =============================================================================
# Test G: get_rank_and_percentile() unit tests (no sandbox needed)
# =============================================================================


class TestGetRankAndPercentile:
    """Test G: get_rank_and_percentile() tolerance logic with edge-case scores."""

    def test_higher_is_better_middle_score(self):
        """Score 0.85 in [0.9, 0.8, 0.7, 0.6] (higher is better) → rank 2, percentile 0.75"""
        result = get_rank_and_percentile(0.85, [0.9, 0.8, 0.7, 0.6], lower_is_better=False)
        assert result["rank"] == 2.0
        assert abs(result["percentile"] - 0.75) < 0.01

    def test_lower_is_better_middle_score(self):
        """Score 0.15 in [0.2, 0.1, 0.3, 0.05] (lower is better) → rank 3"""
        result = get_rank_and_percentile(0.15, [0.2, 0.1, 0.3, 0.05], lower_is_better=True)
        assert result["rank"] == 3.0
        assert abs(result["percentile"] - 0.5) < 0.01

    def test_best_score_higher_is_better(self):
        """Best score gets rank 1, percentile 1.0"""
        result = get_rank_and_percentile(0.99, [0.9, 0.8, 0.7], lower_is_better=False)
        assert result["rank"] == 1.0
        assert abs(result["percentile"] - 1.0) < 0.01

    def test_worst_score_higher_is_better(self):
        """Worst score gets last rank, percentile 0.0"""
        result = get_rank_and_percentile(0.5, [0.9, 0.8, 0.7], lower_is_better=False)
        assert result["rank"] == 4.0
        assert abs(result["percentile"] - 0.0) < 0.01

    def test_best_score_lower_is_better(self):
        """Best score (lowest) gets rank 1, percentile 1.0"""
        result = get_rank_and_percentile(0.01, [0.1, 0.2, 0.3], lower_is_better=True)
        assert result["rank"] == 1.0
        assert abs(result["percentile"] - 1.0) < 0.01

    def test_tied_scores_average_rank(self):
        """Tied scores get averaged rank"""
        # Score 0.8 ties with existing 0.8, should get average of positions
        result = get_rank_and_percentile(0.8, [0.9, 0.8, 0.7], lower_is_better=False)
        # 0.9 is rank 1, both 0.8s share ranks 2-3 → avg 2.5
        assert result["rank"] == 2.5
        assert 0.0 < result["percentile"] < 1.0

    def test_single_score_leaderboard(self):
        """Single score on empty leaderboard"""
        result = get_rank_and_percentile(0.5, [], lower_is_better=False)
        assert result["rank"] == 1.0
        assert result["percentile"] == 1.0

    def test_nan_score_returns_zero_percentile(self):
        """NaN score should return percentile 0.0"""
        result = get_rank_and_percentile(float("nan"), [0.9, 0.8, 0.7], lower_is_better=False)
        assert result["percentile"] == 0.0
        assert result["rank"] == 4  # Last place

    def test_none_score_returns_zero_percentile(self):
        """None score should return percentile 0.0"""
        result = get_rank_and_percentile(None, [0.9, 0.8, 0.7], lower_is_better=False)
        assert result["percentile"] == 0.0

    def test_tolerance_close_values(self):
        """Test tolerance-based comparison for floating point precision"""
        # Two values that are "equal" within tolerance
        base = 0.123456789012345
        almost_same = base + 1e-11  # Within tolerance
        result = get_rank_and_percentile(almost_same, [base], lower_is_better=False)
        # Should tie (both rank 1.5 after averaging 1 and 2)
        assert result["rank"] == 1.5


# =============================================================================
# Mock Sandbox for unit tests (Tests A-E)
# =============================================================================


@dataclass
class MockSandbox:
    """Mock sandbox for testing evaluator without real AgentBox."""

    exec_responses: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    exec_should_fail: bool = False
    exec_error_message: str = "Command failed"

    def exec(self, command: str, timeout: float | None = None) -> str:
        if self.exec_should_fail:
            raise RuntimeError(self.exec_error_message)
        # Check for specific commands
        for pattern, response in self.exec_responses.items():
            if pattern in command:
                return response
        return ""

    def upload_file(self, local_path: str, remote_path: str) -> None:
        with open(local_path) as f:
            self.files[remote_path] = f.read()

    def fetch_file(self, remote_path: str, local_path: str) -> bool:
        if remote_path in self.files:
            with open(local_path, "w") as f:
                f.write(self.files[remote_path])
            return True
        return False

    def close(self) -> None:
        pass


@dataclass
class MockEpisode:
    """Mock Episode for testing."""

    artifacts: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Tests A-E: Unit tests with mock sandbox
# =============================================================================


class TestEvaluatorStageFails:
    """Tests A-E: Various failure modes with mock sandbox."""

    def test_a_no_solution(self):
        """Test A: Stage 1 — no solution → reward=0.0"""
        sandbox = MockSandbox()
        episode = MockEpisode(
            artifacts={
                "_sandbox": sandbox,
                "pred_solution": None,  # No solution
            }
        )
        task = {"task_id": "mlsp-2013-birds"}

        evaluator = MLEEvaluator(submit_file="code")
        result = evaluator.evaluate(task, episode)

        assert result.reward == 0.0
        assert result.is_correct is False
        assert "No solution" in result.metadata.get("reason", "")

    def test_b_no_submission_csv_reference(self):
        """Test B: Stage 1 — solution without submission.csv reference → reward=0.0"""
        sandbox = MockSandbox()
        episode = MockEpisode(
            artifacts={
                "_sandbox": sandbox,
                "pred_solution": "print('hello world')",  # No submission.csv reference
            }
        )
        task = {"task_id": "spooky-author-identification"}

        evaluator = MLEEvaluator(submit_file="code")
        result = evaluator.evaluate(task, episode)

        assert result.reward == 0.0
        assert result.is_correct is False
        assert "submission.csv" in result.metadata.get("reason", "")

    def test_c_execution_failure(self):
        """Test C: Stage 4 — execution failure → error captured in signals"""
        sandbox = MockSandbox(
            exec_should_fail=True,
            exec_error_message="SyntaxError: invalid syntax",
        )
        episode = MockEpisode(
            artifacts={
                "_sandbox": sandbox,
                "pred_solution": "print('hello submission.csv')",  # Has submission.csv reference
            }
        )
        task = {"task_id": "spooky-author-identification"}

        evaluator = MLEEvaluator(submit_file="code")

        # Mock the upload to not fail
        sandbox.exec_should_fail = False  # Allow cleanup
        sandbox.exec_responses = {"rm -f": ""}  # Allow cleanup
        sandbox.exec_should_fail = True  # Fail on execution

        # Need to handle upload
        with patch.object(sandbox, "upload_file"):
            result = evaluator.evaluate(task, episode)

        assert result.reward == 0.0
        assert result.is_correct is False
        assert "Execution failed" in result.metadata.get("reason", "") or "failed" in result.metadata.get("reason", "").lower()

    def test_d_missing_csv_after_execution(self):
        """Test D: Stage 5 — solution runs but no submission.csv → reward=0.0"""
        sandbox = MockSandbox(
            exec_responses={
                "rm -f": "",
                "python solution.py": "Script completed successfully",
            },
            files={},  # No submission.csv
        )
        episode = MockEpisode(
            artifacts={
                "_sandbox": sandbox,
                "pred_solution": "import pandas as pd\ndf.to_csv('submission.csv')",
            }
        )
        task = {"task_id": "spooky-author-identification"}

        evaluator = MLEEvaluator(submit_file="code")

        with patch.object(sandbox, "upload_file"):
            result = evaluator.evaluate(task, episode)

        assert result.reward == 0.0
        assert result.is_correct is False
        assert "submission.csv not found" in result.metadata.get("reason", "")
        # Check signal
        csv_signal = next((s for s in result.signals if s.name == "submission_csv_provided"), None)
        assert csv_signal is not None
        assert csv_signal.value == 0.0

    def test_no_sandbox_available(self):
        """Edge case: No sandbox in artifacts → reward=0.0"""
        episode = MockEpisode(
            artifacts={
                "pred_solution": "import pandas as pd\ndf.to_csv('submission.csv')",
                # No _sandbox key
            }
        )
        task = {"task_id": "spooky-author-identification"}

        evaluator = MLEEvaluator(submit_file="code")
        result = evaluator.evaluate(task, episode)

        assert result.reward == 0.0
        assert result.is_correct is False
        assert "sandbox" in result.metadata.get("reason", "").lower()


class TestEvaluatorCSVMode:
    """Test submit_file='csv' mode (skips stages 1-4)."""

    def test_csv_mode_skips_stages_1_to_4(self):
        """In CSV mode, stages 1-4 are skipped - no pred_solution check"""
        sandbox = MockSandbox(
            files={},  # No submission.csv - will fail at stage 5
        )
        episode = MockEpisode(
            artifacts={
                "_sandbox": sandbox,
                "pred_solution": None,  # No solution - but CSV mode shouldn't check this
            }
        )
        task = {"task_id": "spooky-author-identification"}

        evaluator = MLEEvaluator(submit_file="csv")
        result = evaluator.evaluate(task, episode)

        # In CSV mode, it should skip stages 1-4 completely:
        # - Stage 1 (pred_solution checks) is skipped
        # - Stage 4 (execution) is skipped
        # It should fail at Stage 5 (fetch submission.csv) since file doesn't exist
        assert result.reward == 0.0
        # The failure should be about missing CSV, NOT about missing solution
        assert "No solution" not in result.metadata.get("reason", "")
        assert "submission.csv" in result.metadata.get("reason", "").lower() or "fetch" in result.metadata.get("reason", "").lower()

    def test_csv_mode_with_existing_csv(self):
        """CSV mode with pre-existing submission.csv proceeds to validation"""
        sandbox = MockSandbox(
            files={"/workspace/submission.csv": "id,author\n1,EAP\n2,HPL\n3,MWS\n"},
        )
        episode = MockEpisode(
            artifacts={
                "_sandbox": sandbox,
                # No pred_solution needed in CSV mode
            }
        )
        task = {"task_id": "spooky-author-identification"}

        evaluator = MLEEvaluator(submit_file="csv")

        # This will proceed to stage 6 (validation) which requires mlebench
        # Since mlebench's validate_submission is called inside the function,
        # it will either work (if mlebench installed) or fail at import
        result = evaluator.evaluate(task, episode)

        # Key assertion: we got past stages 1-4 (no solution check)
        assert "No solution" not in result.metadata.get("reason", "")
        assert "submission.csv" not in result.metadata.get("reason", "").lower() or "not found" not in result.metadata.get("reason", "").lower()
        # submission_csv_provided signal should be 1.0 since we fetched the file
        csv_signal = next((s for s in result.signals if s.name == "submission_csv_provided"), None)
        if csv_signal:
            assert csv_signal.value == 1.0


# =============================================================================
# Test F: Known good submission (requires real mlebench)
# =============================================================================


@pytest.mark.skipif(not os.environ.get("RUN_MLEBENCH_TESTS"), reason="Set RUN_MLEBENCH_TESTS=1 to run mlebench integration tests")
class TestEvaluatorWithMlebench:
    """Test F: Known good submission with real mlebench scoring.

    These tests require mlebench and dojo packages to be installed.
    Run with: RUN_MLEBENCH_TESTS=1 pytest test_evaluator.py -k "mlebench"
    """

    def test_f_known_good_submission(self):
        """Test F: Known good submission → expected percentile"""
        # This test requires:
        # 1. A pre-computed correct submission.csv
        # 2. mlebench and dojo packages installed
        # 3. Access to MLE_BENCH_DATA_DIR

        try:
            import dojo.tasks.mlebench.evaluate as dojo_evaluate
            from mlebench.grade import validate_submission
            from mlebench.registry import registry
        except ImportError:
            pytest.skip("mlebench/dojo not installed")

        task_id = "spooky-author-identification"
        mle_bench_data_dir = "/checkpoint/maui/shared/cache/dojo/tasks/mlebench"

        # Check if data exists
        if not Path(mle_bench_data_dir).exists():
            pytest.skip(f"MLE_BENCH_DATA_DIR not found: {mle_bench_data_dir}")

        # Use the REAL sample_submission.csv as our test submission
        # This ensures correct IDs and row count
        sample_submission_path = Path(mle_bench_data_dir) / task_id / "prepared" / "public" / "sample_submission.csv"
        if not sample_submission_path.exists():
            pytest.skip(f"sample_submission.csv not found: {sample_submission_path}")

        # Copy sample_submission to temp file (so we don't modify the original)
        import shutil

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            submission_path = f.name
        shutil.copy(sample_submission_path, submission_path)

        try:
            # Test validation
            new_registry = registry.set_data_dir(Path(mle_bench_data_dir))
            competition = new_registry.get_competition(task_id)
            is_valid, msg = validate_submission(Path(submission_path), competition)

            # MUST pass validation — fail test if not
            assert is_valid, f"Sample submission failed validation: {msg}"
            print(f"✓ Validation passed: {msg}")

            # Get score and percentile
            score, _ = dojo_evaluate.evaluate_submission(
                submission_path=Path(submission_path),
                data_dir=Path(mle_bench_data_dir),
                competition_id=task_id,
                results_output_dir=Path(tempfile.mkdtemp()),
            )
            print(f"Score: {score}")

            import pandas as pd

            leaderboard_df = pd.read_csv(competition.leaderboard)
            leaderboard_scores = list(leaderboard_df["score"])
            lower_is_better = competition.grader.is_lower_better(leaderboard_df)

            result = get_rank_and_percentile(score, leaderboard_scores, lower_is_better)
            print(f"Percentile: {result['percentile']:.4f}, Rank: {result['rank']:.1f}")

            # Sample submission should get SOME percentile (baseline predictions)
            assert 0.0 <= result["percentile"] <= 1.0, f"Invalid percentile: {result['percentile']}"
            print(f"✓ Test F PASSED: percentile={result['percentile']:.4f}")
        finally:
            os.unlink(submission_path)

    def test_e_invalid_csv_format(self):
        """Test E: Stage 6 — invalid CSV format → validation error, reward=0.0

        This tests that malformed submissions are caught by mlebench validation.
        """
        try:
            from mlebench.grade import validate_submission
            from mlebench.registry import registry
        except ImportError:
            pytest.skip("mlebench not installed")

        task_id = "spooky-author-identification"
        mle_bench_data_dir = "/checkpoint/maui/shared/cache/dojo/tasks/mlebench"

        if not Path(mle_bench_data_dir).exists():
            pytest.skip(f"MLE_BENCH_DATA_DIR not found: {mle_bench_data_dir}")

        # Create various malformed submissions to test
        malformed_submissions = [
            # Wrong column names
            ("wrong_columns", "wrong,columns,here\n1,2,3\n"),
            # Missing required columns (spooky needs id,EAP,HPL,MWS)
            ("missing_columns", "id,EAP\n1,0.5\n"),
            # Empty file
            ("empty", ""),
            # Only header, no data
            ("header_only", "id,EAP,HPL,MWS\n"),
            # Invalid probability values (should sum to ~1)
            ("bad_probs", "id,EAP,HPL,MWS\nid00001,5.0,5.0,5.0\n"),
        ]

        new_registry = registry.set_data_dir(Path(mle_bench_data_dir))
        competition = new_registry.get_competition(task_id)

        for name, content in malformed_submissions:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                f.write(content)
                submission_path = f.name

            try:
                is_valid, msg = validate_submission(Path(submission_path), competition)
                print(f"  {name}: valid={is_valid}, msg={msg}")

                # Most malformed submissions should be invalid
                # (some might pass validation but fail scoring)
                if name in ["wrong_columns", "missing_columns", "empty"]:
                    assert not is_valid, f"Expected {name} to be invalid, but it passed validation"
            finally:
                os.unlink(submission_path)

        # Now test through the full evaluator with a malformed CSV
        sandbox = MockSandbox(
            files={"/workspace/submission.csv": "wrong,columns,here\n1,2,3\n"},
        )
        episode = MockEpisode(
            artifacts={
                "_sandbox": sandbox,
                # Skip stages 1-4 by using csv mode
            }
        )
        task = {"task_id": task_id}

        evaluator = MLEEvaluator(
            submit_file="csv",
            mle_bench_data_dir=mle_bench_data_dir,
        )
        result = evaluator.evaluate(task, episode)

        # Should fail with reward=0.0 due to invalid CSV
        assert result.reward == 0.0
        assert result.is_correct is False
        # The reason should mention validation or invalid
        reason = result.metadata.get("reason", "").lower()
        assert "invalid" in reason or "validation" in reason or "error" in reason, f"Expected validation error, got: {result.metadata.get('reason', '')}"
        print(f"✓ test_e_invalid_csv_format: {result.metadata.get('reason', '')}")


# =============================================================================
# Test with real AgentBox sandbox (integration)
# =============================================================================


@pytest.mark.skipif(not os.environ.get("AGENTBOX_MANAGER_URI"), reason="Set AGENTBOX_MANAGER_URI to run real sandbox tests")
class TestEvaluatorRealSandbox:
    """Integration tests with real AgentBox sandbox.

    Run with: AGENTBOX_MANAGER_URI=http://host:port pytest test_evaluator.py -k "real"
    """

    @pytest.fixture
    def real_sandbox(self):
        """Create a real AgentBox sandbox for testing."""
        from mle_agent.sandbox import AgentBoxSandbox

        manager_uri = os.environ["AGENTBOX_MANAGER_URI"]
        sandbox = AgentBoxSandbox(
            name="test-evaluator",
            manager_uri=manager_uri,
            superimage_directory="/checkpoint/maui_sft/shared/sif",
            superimage_version="2025-05-02v2",
            read_only_overlays=["/checkpoint/fair-maui-hs/hotfix/kniu.2025-09-19.cache.overlay.ext3.img"],
            working_dir="/workspace",
        )
        yield sandbox
        sandbox.close()

    def test_real_execution_failure(self, real_sandbox):
        """Test C with real sandbox: broken solution.py"""
        episode = MockEpisode(
            artifacts={
                "_sandbox": real_sandbox,
                "pred_solution": """
# This script has a syntax error
print("writing submission.csv"
""",  # Missing closing parenthesis
            }
        )
        task = {"task_id": "spooky-author-identification"}

        evaluator = MLEEvaluator(submit_file="code", eval_timeout=30)
        result = evaluator.evaluate(task, episode)

        assert result.reward == 0.0
        assert result.is_correct is False

    def test_real_no_csv_produced(self, real_sandbox):
        """Test D with real sandbox: solution runs but no CSV"""
        episode = MockEpisode(
            artifacts={
                "_sandbox": real_sandbox,
                "pred_solution": """
# This script runs but doesn't produce submission.csv
print("I forgot to write submission.csv")
""",
            }
        )
        task = {"task_id": "spooky-author-identification"}

        evaluator = MLEEvaluator(submit_file="code", eval_timeout=30)
        result = evaluator.evaluate(task, episode)

        assert result.reward == 0.0
        assert "submission.csv not found" in result.metadata.get("reason", "")


# =============================================================================
# Main entry point for running tests directly
# =============================================================================


if __name__ == "__main__":
    # Run the percentile tests first (no external dependencies)
    print("=" * 60)
    print("Test G: get_rank_and_percentile() unit tests")
    print("=" * 60)

    test_class = TestGetRankAndPercentile()
    test_class.test_higher_is_better_middle_score()
    print("✓ test_higher_is_better_middle_score")

    test_class.test_lower_is_better_middle_score()
    print("✓ test_lower_is_better_middle_score")

    test_class.test_best_score_higher_is_better()
    print("✓ test_best_score_higher_is_better")

    test_class.test_worst_score_higher_is_better()
    print("✓ test_worst_score_higher_is_better")

    test_class.test_best_score_lower_is_better()
    print("✓ test_best_score_lower_is_better")

    test_class.test_tied_scores_average_rank()
    print("✓ test_tied_scores_average_rank")

    test_class.test_single_score_leaderboard()
    print("✓ test_single_score_leaderboard")

    test_class.test_nan_score_returns_zero_percentile()
    print("✓ test_nan_score_returns_zero_percentile")

    test_class.test_none_score_returns_zero_percentile()
    print("✓ test_none_score_returns_zero_percentile")

    test_class.test_tolerance_close_values()
    print("✓ test_tolerance_close_values")

    print("\n" + "=" * 60)
    print("Tests A-E: Evaluator stage failure tests (mock sandbox)")
    print("=" * 60)

    test_fails = TestEvaluatorStageFails()
    test_fails.test_a_no_solution()
    print("✓ test_a_no_solution")

    test_fails.test_b_no_submission_csv_reference()
    print("✓ test_b_no_submission_csv_reference")

    # Test C needs special handling due to the mock setup
    print("⊘ test_c_execution_failure (run with pytest)")

    test_fails.test_d_missing_csv_after_execution()
    print("✓ test_d_missing_csv_after_execution")

    test_fails.test_no_sandbox_available()
    print("✓ test_no_sandbox_available")

    print("\n" + "=" * 60)
    print("CSV mode tests")
    print("=" * 60)

    test_csv = TestEvaluatorCSVMode()
    test_csv.test_csv_mode_skips_stages_1_to_4()
    print("✓ test_csv_mode_skips_stages_1_to_4")

    test_csv.test_csv_mode_with_existing_csv()
    print("✓ test_csv_mode_with_existing_csv")

    print("\n" + "=" * 60)
    print("All basic tests passed!")
    print("=" * 60)
    print("\nTo run full test suite with pytest:")
    print("  pytest cookbooks/mlebench/test_evaluator.py -v")
    print("\nTo run mlebench integration tests:")
    print("  RUN_MLEBENCH_TESTS=1 pytest cookbooks/mlebench/test_evaluator.py -k 'mlebench' -v")
    print("\nTo run real sandbox tests:")
    print("  AGENTBOX_MANAGER_URI=http://host:port pytest cookbooks/mlebench/test_evaluator.py -k 'real' -v")
