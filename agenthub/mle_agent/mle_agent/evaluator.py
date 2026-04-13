"""MLE-bench evaluator.

This module implements the MLEEvaluator class which scores
agent submissions against Kaggle leaderboards using the 7-stage
evaluation pipeline from amaia-collab.

Stages:
1. Sanity checks - verify solution exists and mentions submission.csv
2. Setup - create temp directory, prepare paths
3. Upload - write solution to container as /workspace/solution.py
4. Execute - run the solution script with timeout
5. Fetch - retrieve /workspace/submission.csv from container
6. Validate - check submission format matches competition requirements
7. Score - evaluate submission against ground truth and compute percentile
"""

from __future__ import annotations

import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Any

from rllm.experimental.eval.types import EvalOutput, Signal
from rllm.types import Episode

logger = logging.getLogger(__name__)

# Default path matching AMAIA
MLE_BENCH_DATA_DIR = "/checkpoint/maui/shared/cache/dojo/tasks/mlebench"


def get_rank_and_percentile(score: float, leaderboard_scores: list[float], lower_is_better: bool) -> dict[str, float | None]:
    """
    Calculates the percentile rank of `score` as if it were an additional submission in the leaderboard.

    The function computes the average rank of `score` among all scores (including itself) and then maps it
    to a percentile between 0 and 1 using:

        percentile = (n - avg_rank) / (n - 1)

    where n is the total number of scores after including the new one.

    - A percentile of 1 indicates the best score.
    - A percentile of 0 indicates the worst score.

    Ported from amaia-collab/apps/sea/envs/envs/mle_bench/evaluation.py
    """
    import numpy as np

    # Return early if score is None or NaN
    if score is None or np.isnan(score):
        return {"percentile": 0.0, "rank": len(leaderboard_scores) + 1}

    # Combine existing scores with the new score
    scores_list = list(leaderboard_scores) + [score]
    n = len(scores_list)

    # Handle edge case: only one score
    if n == 1:
        return {"percentile": 1.0, "rank": 1.0}

    # Sort scores: best first. For lower_is_better, lower numbers are better.
    if lower_is_better:
        sorted_scores = sorted(scores_list)  # ascending: best is first
    else:
        sorted_scores = sorted(scores_list, reverse=True)  # descending: best is first

    # Use a tolerance-based comparison to handle floating point imprecision.
    # Find all positions (1-indexed) where the new score is "close" to an existing score.
    tol_rel = 1e-9
    tol_abs = 1e-12
    ranks = [i + 1 for i, s in enumerate(sorted_scores) if math.isclose(s, score, rel_tol=tol_rel, abs_tol=tol_abs)]

    # In case no value is considered close (shouldn't happen because score is in scores_list), fall back to exact equality.
    if not ranks:
        ranks = [i + 1 for i, s in enumerate(sorted_scores) if s == score]

    avg_rank = sum(ranks) / len(ranks)

    # Compute the percentile
    percentile = (n - avg_rank) / (n - 1)

    return {"percentile": percentile, "rank": avg_rank}


def _fail(message: str, task_id: str, signals: list[Signal] | None = None, **metadata: Any) -> EvalOutput:
    """Helper to create a failed EvalOutput."""
    # Start with default signals
    signal_dict = {
        "percentile": 0.0,
        "valid_submission": 0.0,
        "submission_csv_provided": 0.0,
    }
    # Override with passed signals
    if signals:
        for sig in signals:
            signal_dict[sig.name] = sig.value

    all_signals = [Signal(name, value) for name, value in signal_dict.items()]

    return EvalOutput(
        reward=0.0,
        is_correct=False,
        signals=all_signals,
        metadata={"task_id": task_id, "reason": message, **metadata},
    )


class MLEEvaluator:
    """Evaluator for MLE-bench tasks using 7-stage grading pipeline.

    Args:
        eval_timeout: Timeout in seconds for solution execution (Stage 4)
        mle_bench_data_dir: Path to MLE-bench data directory containing competition data
        submit_file: "code" to run solution.py and generate CSV,
                     "csv" to skip stages 1-4 and fetch existing submission.csv directly
    """

    def __init__(
        self,
        eval_timeout: int = 300,
        mle_bench_data_dir: str = MLE_BENCH_DATA_DIR,
        submit_file: str = "code",
    ):
        self.eval_timeout = eval_timeout
        self.mle_bench_data_dir = Path(mle_bench_data_dir)
        self.submit_file = submit_file

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        """Run 7-stage evaluation pipeline.

        Accesses the sandbox from episode.artifacts["_sandbox"] which is
        injected by EvalRunner after the agent's run() completes.
        """
        # Get sandbox from artifacts (injected by EvalRunner)
        sandbox = episode.artifacts.get("_sandbox")
        task_id = task.get("task_id") or task.get("instance_id", "unknown")
        pred_solution = episode.artifacts.get("pred_solution")

        if sandbox is None:
            return _fail("No sandbox available", task_id)

        # Create temp directory for local files
        temp_dir = Path(tempfile.mkdtemp(prefix="mlebench_eval_", dir="/tmp"))
        submission_file_path = temp_dir / "submission.csv"

        # Initialize execution tracking
        execution_time = 0.0
        exec_output = ""

        try:
            # =================================================================
            # STAGES 1-4: Only when submit_file="code"
            # When submit_file="csv", skip to stage 5
            # =================================================================
            if self.submit_file == "code":
                # Stage 1: Sanity checks
                logger.info(
                    f"[mlebench eval] task={task_id}, STAGE 1: Sanity checks. pred_solution is None: {pred_solution is None}, pred_solution length: {len(pred_solution) if pred_solution else 0}"
                )

                if not pred_solution:
                    return _fail("No solution submitted", task_id)

                if "submission.csv" not in pred_solution:
                    return _fail(
                        "Solution does not reference submission.csv",
                        task_id,
                        solution_preview=pred_solution[:500] if pred_solution else "",
                    )

                # Stage 2: Setup - remove any stale submission.csv
                try:
                    sandbox.exec("rm -f /workspace/submission.csv")
                except Exception as e:
                    logger.warning(f"[mlebench eval] task={task_id}, cleanup failed: {e}")

                # Stage 3: Upload solution to container
                solution_local_path = temp_dir / "solution.py"
                with open(solution_local_path, "w") as f:
                    f.write(pred_solution)

                try:
                    sandbox.upload_file(str(solution_local_path), "/workspace/solution.py")
                except Exception as e:
                    return _fail(f"Failed to upload solution: {e}", task_id)

                # Stage 4: Execute solution
                logger.info(f"[mlebench eval] task={task_id}, STAGE 4: Executing solution")
                start_time = time.time()

                try:
                    exec_output = sandbox.exec(
                        "cd /workspace && python solution.py",
                        timeout=float(self.eval_timeout),
                    )
                    execution_time = time.time() - start_time
                except Exception as e:
                    execution_time = time.time() - start_time
                    error_msg = str(e)
                    if "timeout" in error_msg.lower():
                        return _fail(
                            "Execution timed out",
                            task_id,
                            signals=[Signal("exec_duration", execution_time)],
                            exec_output=exec_output[:2000] if exec_output else "",
                        )
                    return _fail(
                        f"Execution failed: {error_msg}",
                        task_id,
                        signals=[Signal("exec_duration", execution_time)],
                        exec_output=exec_output[:2000] if exec_output else "",
                    )

                logger.info(f"[mlebench eval] task={task_id}, execution completed in {execution_time:.2f}s")

            else:
                # submit_file == "csv": Skip stages 1-4
                logger.info(f"[mlebench eval] task={task_id}, submit_file=csv, skipping stages 1-4")

            # =================================================================
            # STAGE 5: Fetch submission.csv from container
            # =================================================================
            logger.info(f"[mlebench eval] task={task_id}, STAGE 5: Fetching submission.csv")

            try:
                has_csv = sandbox.fetch_file("/workspace/submission.csv", str(submission_file_path))
            except Exception as e:
                return _fail(
                    f"Failed to fetch submission.csv: {e}",
                    task_id,
                    exec_output=exec_output[:2000] if exec_output else "",
                )

            if not has_csv or not submission_file_path.exists():
                return _fail(
                    "submission.csv not found after execution",
                    task_id,
                    signals=[Signal("submission_csv_provided", 0.0)],
                    exec_output=exec_output[:2000] if exec_output else "",
                )

            # =================================================================
            # STAGE 6: Validate submission format
            # =================================================================
            logger.info(f"[mlebench eval] task={task_id}, STAGE 6: Validating submission")

            try:
                from mlebench.grade import validate_submission
                from mlebench.registry import registry

                # Set data directory and get competition
                new_registry = registry.set_data_dir(self.mle_bench_data_dir)
                competition = new_registry.get_competition(task_id)

                is_valid, validation_msg = validate_submission(submission_file_path, competition)

                if not is_valid:
                    # Read submission preview for debugging
                    try:
                        with open(submission_file_path) as f:
                            submission_preview = f.read(2000)
                    except Exception:
                        submission_preview = "<could not read submission>"

                    return _fail(
                        f"Invalid submission: {validation_msg}",
                        task_id,
                        signals=[Signal("submission_csv_provided", 1.0)],
                        submission_preview=submission_preview[:500],
                    )

            except ImportError:
                logger.warning(f"[mlebench eval] task={task_id}, mlebench not installed, skipping validation")
                competition = None

            # =================================================================
            # STAGE 7: Score submission and compute percentile
            # =================================================================
            logger.info(f"[mlebench eval] task={task_id}, STAGE 7: Scoring submission")

            try:
                import dojo.tasks.mlebench.evaluate as dojo_evaluate
                import pandas as pd

                # Evaluate submission
                score, _ = dojo_evaluate.evaluate_submission(
                    submission_path=submission_file_path,
                    data_dir=self.mle_bench_data_dir,
                    competition_id=task_id,
                    results_output_dir=temp_dir,
                )

                # Get leaderboard and compute percentile
                if competition is not None:
                    leaderboard_df = pd.read_csv(competition.leaderboard)
                    leaderboard_scores = list(leaderboard_df["score"])
                    lower_is_better = competition.grader.is_lower_better(leaderboard_df)
                else:
                    # Fallback if competition object not available
                    leaderboard_scores = []
                    lower_is_better = True

                result = get_rank_and_percentile(
                    score=score,
                    leaderboard_scores=leaderboard_scores,
                    lower_is_better=lower_is_better,
                )
                percentile = result["percentile"]
                rank = result["rank"]

            except ImportError as e:
                return _fail(
                    f"dojo/mlebench not installed: {e}",
                    task_id,
                    signals=[Signal("submission_csv_provided", 1.0)],
                )
            except Exception as e:
                logger.exception(f"[mlebench eval] task={task_id}, scoring error")
                return _fail(
                    f"Scoring error: {e}",
                    task_id,
                    signals=[Signal("submission_csv_provided", 1.0)],
                )

            # =================================================================
            # SUCCESS
            # =================================================================
            logger.info(f"[mlebench eval] task={task_id}, SUCCESS: score={score}, percentile={percentile:.4f}, rank={rank}, execution_time={execution_time:.2f}s")

            return EvalOutput(
                reward=percentile,
                is_correct=percentile > 0.0,
                signals=[
                    Signal("percentile", percentile),
                    Signal("raw_score", float(score) if score is not None else 0.0),
                    Signal("rank", float(rank) if rank is not None else 0.0),
                    Signal("valid_submission", 1.0),
                    Signal("submission_csv_provided", 1.0),
                    Signal("exec_duration", execution_time),
                ],
                metadata={
                    "task_id": task_id,
                    "score": score,
                    "percentile": percentile,
                    "rank": rank,
                },
            )

        finally:
            # Cleanup temp directory
            try:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
