"""Evaluation utilities for analyzing mlebench evaluation results."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

# Try to import mlebench components
try:
    from mlebench.registry import registry

    MLEBENCH_AVAILABLE = True
except ImportError:
    registry = None
    MLEBENCH_AVAILABLE = False


# Define infrastructure error categories
INFRA_ERROR_CATEGORIES = {
    "Infrastructure: Connection/Heartbeat Error",
    "Infrastructure: Container Error",
    "Infrastructure: Container Execution Error",
    "Infrastructure: Container Creation Error",
    "Infrastructure: Model Call Error",
    "Infrastructure: Too many streaming blocks of output",
}


def is_infra_error_category(category: str) -> bool:
    """Check if an error category is an infrastructure error."""
    return category in INFRA_ERROR_CATEGORIES


def categorize_error(row):
    """Categorize error based on common patterns in the error message and infrastructure flags.

    Adapted for the new mlebench trajectories structure where outcomes are nested in 'outcomes' dict.

    Args:
        row: A dict-like object. Can contain either:
            - Nested 'outcomes' dict with error flags, or
            - Flat columns with error flags directly
    """
    # Handle both nested (from df with 'outcomes' column) and flat structures
    if "outcomes" in row and isinstance(row.get("outcomes"), dict):
        outcomes = row["outcomes"]
    else:
        outcomes = row

    # Use both error message and output for categorization
    error_output = outcomes.get("eval_error_output", "")
    error_message = outcomes.get("eval_error_message", "")
    # Combine both for pattern matching - message is the primary reason, output is execution details
    error_str = str(error_output).lower() + " " + str(error_message).lower()
    error_str_original = str(error_output) + " " + str(error_message)

    model_call_error = outcomes.get("model_call_error", False)
    max_turns_reached = outcomes.get("max_turns_reached", False)
    rollout_timeout = outcomes.get("rollout_timeout", False)
    parse_error = outcomes.get("parse_error", False)
    context_exceeded = outcomes.get("context_exceeded", False)
    eval_timeout = outcomes.get("eval_timeout", False)

    # Check infrastructure errors first
    if model_call_error:
        return "Infrastructure: Model Call Error"

    if context_exceeded:
        return "Context Length Exceeded"

    if max_turns_reached:
        return "Max Turns Reached"

    if rollout_timeout:
        return "Rollout Timeout"

    if eval_timeout:
        return "Eval Timeout"

    if parse_error:
        return "Tool Call Parsing Error"

    # Check if both error_message and error_output are empty (combined string would just be a space)
    if error_str.strip() == "" or error_str.strip() == "none":
        return "Empty error message"

    # Check for submission.csv not found (common in code mode)
    if "submission.csv not found" in error_str:
        return "submission.csv Not Generated"

    # Check for common error patterns
    if "worker connection failed" in error_str or "worker became unresponsive" in error_str:
        return "Infrastructure: Connection/Heartbeat Error"
    elif "heartbeat" in error_str and ("fail" in error_str or "timeout" in error_str):
        return "Infrastructure: Connection/Heartbeat Error"
    elif "socket closed" in error_str or "grpc" in error_str:
        return "Infrastructure: Connection/Heartbeat Error"

    # OOM Kill
    if "Killed" in error_str_original and ("killed\n" in error_str or error_str.strip().endswith("killed")):
        return "OOM Killed (Linux OOM Killer)"
    elif error_str.strip() == "killed" or error_str.strip().startswith("killed\n"):
        return "OOM Killed (Linux OOM Killer)"
    elif "is killed by signal" in error_str or "killed by signal: killed" in error_str:
        return "OOM Killed (Linux OOM Killer)"

    # Validation errors
    if "validation error" in error_str and "submission invalid" in error_str:
        return "Submission Validation Error (Grading)"
    elif "the set of" in error_str and "must match" in error_str:
        return "Submission ID Mismatch"
    if "submission.csv not in solution" in error_str:
        return "Missing submission.csv in solution"
    if "/workspace/submission.csv: no such file or directory" in error_str:
        return "csv_not_found"

    # Standard error patterns
    if "no solution found" in error_str or "no such file" in error_str:
        return "SolutionNotFoundError"
    if "filenotfounderror" in error_str or "no such file" in error_str:
        return "FileNotFoundError"
    elif "error tokenizing data" in error_str and "c error:" in error_str:
        return "CSV Tokenization Error"
    elif "too many streaming blocks" in error_str:
        return "Infrastructure: Too many streaming blocks of output"
    elif "timeout" in error_str:
        return "Timeout"
    elif "submission.csv" in error_str and ("not found" in error_str or "missing" in error_str or "does not exist" in error_str):
        return "Missing submission.csv"
    elif "submission and answers should have" in error_str:
        return "Wrong Submission Format"
    elif "submission and answers have different lengths" in error_str:
        return "Submission/Answers Length Mismatch"
    elif "memoryerror" in error_str or "out of memory" in error_str or "oom" in error_str:
        return "MemoryError"
    elif "valueerror" in error_str:
        return "ValueError"
    elif "systemexit:" in error_str or "systemexit(" in error_str:
        return "SystemExit"
    elif "syntaxerror" in error_str:
        return "SyntaxError"
    elif "importerror" in error_str or "modulenotfounderror" in error_str:
        return "ImportError"
    elif "keyerror" in error_str:
        return "KeyError"
    elif "typeerror" in error_str:
        return "TypeError"
    elif "indexerror" in error_str:
        return "IndexError"
    elif "attributeerror" in error_str:
        return "AttributeError"
    elif "runtimeerror" in error_str:
        return "RuntimeError"
    elif "zerodivisionerror" in error_str:
        return "ZeroDivisionError"
    elif "permissionerror" in error_str or "permission denied" in error_str:
        return "PermissionError"
    elif "connectionerror" in error_str or "connection refused" in error_str:
        return "ConnectionError"
    elif "shape" in error_str and ("mismatch" in error_str or "different" in error_str):
        return "Shape Mismatch"
    elif "column" in error_str and ("missing" in error_str or "not found" in error_str):
        return "Missing Column"
    elif "invalid" in error_str and "format" in error_str:
        return "Invalid Format"
    else:
        return "Other"


def flatten_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the nested 'outcomes' dict into top-level columns.

    Args:
        df: DataFrame with 'outcomes' column containing nested dicts

    Returns:
        DataFrame with outcomes flattened into columns
    """
    if "outcomes" not in df.columns:
        return df

    outcomes_df = pd.json_normalize(df["outcomes"])
    # Prefix columns to avoid conflicts
    outcomes_df.columns = [f"outcomes_{col}" for col in outcomes_df.columns]

    # Add commonly used shortcuts
    if "outcomes_valid_submission" in outcomes_df.columns:
        outcomes_df["valid_submission"] = outcomes_df["outcomes_valid_submission"]
    if "outcomes_pass" in outcomes_df.columns:
        outcomes_df["pass"] = outcomes_df["outcomes_pass"]

    # Combine with original df (drop original outcomes column)
    result = pd.concat([df.drop(columns=["outcomes"]), outcomes_df], axis=1)
    return result


def plot_submission_validity_breakdown(df, figsize=(6, 4), save_path=None, title=None):
    """
    Create a bar chart showing the breakdown of valid vs invalid vs infrastructure error submissions.

    Adapted for mlebench trajectories structure where outcomes are nested in 'outcomes' dict.

    Args:
        df: DataFrame containing either:
            - 'outcomes' column with nested dict containing 'valid_submission', or
            - 'valid_submission' column directly
        figsize: Tuple specifying figure size (width, height)
        save_path: Path to save the plot. If None, plot is displayed.
        title: Custom title for the plot

    Returns:
        tuple: (filtered_valid, filtered_invalid, filtered_infra) DataFrames for further analysis
    """
    # Extract valid_submission from nested outcomes if needed
    if "outcomes" in df.columns and "valid_submission" not in df.columns:
        df = df.copy()
        df["valid_submission"] = df["outcomes"].apply(lambda x: x.get("valid_submission", False) if isinstance(x, dict) else False)

    # Get error category for each row
    def get_error_category(row):
        return categorize_error(row)

    # Categorize each row and check if it's an infrastructure error
    error_categories = df.apply(get_error_category, axis=1)
    infra_mask = error_categories.apply(is_infra_error_category)

    # Filter data into 3 categories
    filtered_infra = df[infra_mask]
    filtered_valid = df[(df["valid_submission"] == True) & (~infra_mask)]
    filtered_invalid = df[(df["valid_submission"] == False) & (~infra_mask)]

    print(f"Found {len(filtered_valid)} rollouts matching valid criteria")
    print(f"Found {len(filtered_invalid)} rollouts matching invalid criteria (excluding infra errors)")
    print(f"Found {len(filtered_infra)} rollouts with infrastructure errors")

    # If there are infrastructure errors, print a breakdown table by category
    if len(filtered_infra) > 0:
        infra_categories = error_categories[infra_mask]
        infra_category_counts = infra_categories.value_counts().reset_index()
        infra_category_counts.columns = ["Infrastructure Error Category", "Count"]
        infra_category_counts["Percentage"] = (infra_category_counts["Count"] / len(filtered_infra) * 100).round(1)
        infra_category_counts = infra_category_counts.sort_values("Count", ascending=False).reset_index(drop=True)

        print("\nInfrastructure Error Breakdown:")
        print(infra_category_counts.to_string(index=False))

    # Prepare data
    valid_count = len(filtered_valid)
    invalid_count = len(filtered_invalid)
    infra_count = len(filtered_infra)
    total_count = valid_count + invalid_count + infra_count
    eval_total = valid_count + invalid_count

    # Create plot with 3 bars
    fig, ax = plt.subplots(figsize=figsize)
    categories = ["Valid", "Invalid\n(Eval Errors)", "Infrastructure\nErrors"]
    counts = [valid_count, invalid_count, infra_count]
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]  # Green, Red, Orange
    bars = ax.bar(categories, counts, color=colors)

    percentages = [valid_count / eval_total * 100 if eval_total > 0 else 0, invalid_count / eval_total * 100 if eval_total > 0 else 0, infra_count / total_count * 100 if total_count > 0 else 0]
    for bar, count, pct in zip(bars, counts, percentages, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Number of Submissions", fontsize=12)
    ax.set_xlabel("Submission Status", fontsize=12)

    plot_title = title or "Submission Breakdown: Valid vs Invalid vs Infrastructure Errors"
    ax.set_title(plot_title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.25 if max(counts) > 0 else 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {save_path}")
        plt.close()
    else:
        plt.show()

    return filtered_valid, filtered_invalid, filtered_infra


def analyze_invalid_submissions(df):
    """
    Comprehensive analysis of invalid submissions including error categorization and summary.

    Adapted for mlebench trajectories structure.

    Args:
        df: DataFrame with 'outcomes' column containing nested dicts, or flattened DataFrame

    Returns:
        tuple: (df_errors, summary_df, rollout_indices_df) for further analysis
    """
    # Extract valid_submission from nested outcomes if needed
    if "outcomes" in df.columns and "valid_submission" not in df.columns:
        df = df.copy()
        df["valid_submission"] = df["outcomes"].apply(lambda x: x.get("valid_submission", False) if isinstance(x, dict) else False)

    # Filter to only invalid submissions
    invalid_mask = df["valid_submission"] == False
    invalid_df = df[invalid_mask].copy()
    invalid_iloc_positions = np.where(invalid_mask)[0]

    print(f"Analyzing {len(df)} rows for invalid submissions...")
    print(f"Found {len(invalid_df)} invalid submissions")

    # Build the errors dataframe
    eval_errors = []
    for iloc_pos, (idx, row) in zip(invalid_iloc_positions, invalid_df.iterrows(), strict=False):
        outcomes = row.get("outcomes", {}) if "outcomes" in row else row
        if not isinstance(outcomes, dict):
            outcomes = {}

        eval_errors.append(
            {
                "index": iloc_pos,
                "id": row.get("id", ""),
                "task": row.get("task", ""),
                "eval_error_output": outcomes.get("eval_error_output", ""),
                "eval_error_message": outcomes.get("eval_error_message", ""),
                "pred_solution_provided": outcomes.get("pred_solution_provided", None),
                "eval_outcome": outcomes.get("eval_outcome", None),
                "model_call_error": outcomes.get("model_call_error", False),
                "max_turns_reached": outcomes.get("max_turns_reached", False),
                "rollout_timeout": outcomes.get("rollout_timeout", False),
                "parse_error": outcomes.get("parse_error", False),
                "context_exceeded": outcomes.get("context_exceeded", False),
                "eval_timeout": outcomes.get("eval_timeout", False),
            }
        )

    df_errors = pd.DataFrame(eval_errors)

    if len(df_errors) == 0:
        print("No invalid submissions found!")
        return df_errors, pd.DataFrame(), pd.DataFrame()

    # Apply categorization
    df_errors["error_category"] = df_errors.apply(lambda row: categorize_error({"outcomes": row.to_dict()}), axis=1)

    # Display error category distribution
    error_counts = df_errors["error_category"].value_counts()
    error_pct = (error_counts / error_counts.sum() * 100).round(2)

    summary_df = pd.DataFrame({"Count": error_counts, "Percentage": error_pct})
    summary_df.index.name = "Error Category"
    print("\n" + "=" * 60)
    print("ERROR CATEGORY DISTRIBUTION")
    print("=" * 60)
    display(summary_df)
    print(f"\nTotal: {error_counts.sum()}")

    # Create table with rollout indices for each category
    rollout_indices_data = []
    for category in error_counts.index:
        category_df = df_errors[df_errors["error_category"] == category]
        indices = category_df["index"].tolist()
        rollout_indices_data.append({"Error Category": category, "Count": len(indices), "Rollout Indices": str(indices)})

    rollout_indices_df = pd.DataFrame(rollout_indices_data)
    print("\n" + "=" * 60)
    print("ROLLOUT INDICES BY ERROR CATEGORY")
    print("=" * 60)
    display(rollout_indices_df)

    return df_errors, summary_df, rollout_indices_df


def plot_invalid_error_distribution(df_errors, figsize=(12, 6), task_names=None, task_col="task"):
    """
    Visualize the distribution of error types in invalid submissions.

    Args:
        df_errors: DataFrame with 'error_category' column containing categorized errors
        figsize: Tuple specifying figure size (width, height)
        task_names: Optional list of task names to filter on. If None, use all tasks.
        task_col: Column name for task identifier (default: "task")

    Returns:
        tuple: (error_counts, error_pct) for further analysis
    """
    # Filter by task_names if provided
    if task_names is not None:
        df_errors = df_errors[df_errors[task_col].isin(task_names)]
        print(f"Filtered to {len(df_errors)} errors for {len(task_names)} tasks")

    # Calculate error distribution
    error_counts = df_errors["error_category"].value_counts()
    error_pct = (error_counts / error_counts.sum() * 100).round(2)

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=figsize)
    error_counts.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Count")
    ax.set_ylabel("Error Category")

    # Set title based on whether task_names is provided
    if task_names is not None:
        if len(task_names) == 1:
            title = f"Distribution of Error Types in Invalid Submissions\nTask: {task_names[0]}"
        else:
            title = f"Distribution of Error Types in Invalid Submissions\n({len(task_names)} tasks)"
    else:
        title = "Distribution of Error Types in Invalid Submissions"
    ax.set_title(title)

    # Add percentage labels on each bar
    total = error_counts.sum()
    for i, (count, category) in enumerate(zip(error_counts.values, error_counts.index, strict=False)):
        pct = count / total * 100
        ax.text(count + 0.5, i, f"{count} ({pct:.1f}%)", va="center", fontsize=9)

    # Adjust x-axis to make room for labels
    ax.set_xlim(0, error_counts.max() * 1.25)

    plt.tight_layout()
    plt.show()

    # Show percentage breakdown
    print("\n" + "=" * 60)
    print("ERROR CATEGORY PERCENTAGE")
    print("=" * 60)
    for cat, pct in error_pct.items():
        print(f"{cat}: {pct}%")

    return error_counts, error_pct


def get_summary_stats(df):
    """Get summary statistics from a trajectories DataFrame.

    Args:
        df: DataFrame loaded from trajectories.jsonl

    Returns:
        dict with summary statistics
    """
    # Extract outcomes
    if "outcomes" in df.columns:
        outcomes_df = pd.json_normalize(df["outcomes"])
    else:
        outcomes_df = df

    # Extract metrics
    if "metrics" in df.columns:
        metrics_df = pd.json_normalize(df["metrics"])
    else:
        metrics_df = df

    total = len(df)
    valid = outcomes_df.get("valid_submission", pd.Series([False] * total)).sum()
    passed = outcomes_df.get("pass", pd.Series([False] * total)).sum()

    stats = {
        "total": total,
        "valid_submissions": int(valid),
        "valid_rate": valid / total * 100 if total > 0 else 0,
        "passed": int(passed),
        "pass_rate": passed / total * 100 if total > 0 else 0,
    }

    if "percentile" in metrics_df.columns:
        percentiles = metrics_df["percentile"].dropna()
        if len(percentiles) > 0:
            stats["avg_percentile"] = percentiles.mean()
            stats["min_percentile"] = percentiles.min()
            stats["max_percentile"] = percentiles.max()

    if "score" in metrics_df.columns:
        scores = metrics_df["score"].dropna()
        if len(scores) > 0:
            stats["avg_score"] = scores.mean()

    return stats


# ======================== MEDAL UTILITIES ========================


def get_rank_and_percentile(score, leaderboard, lower_is_better):
    """
    Calculates the percentile rank of `score` as if it were an additional submission in the leaderboard.

    Args:
        score: The score to rank
        leaderboard: DataFrame with 'score' column
        lower_is_better: If True, lower scores are better

    Returns:
        dict with 'percentile' and 'rank' keys
    """
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return {"percentile": 0, "rank": len(list(leaderboard["score"])) + 1}

    scores_list = list(leaderboard["score"]) + [score]
    n = len(scores_list)

    if lower_is_better:
        sorted_scores = sorted(scores_list)
    else:
        sorted_scores = sorted(scores_list, reverse=True)

    tol_rel = 1e-9
    tol_abs = 1e-12
    ranks = [i + 1 for i, s in enumerate(sorted_scores) if math.isclose(s, score, rel_tol=tol_rel, abs_tol=tol_abs)]

    if not ranks:
        ranks = [i + 1 for i, s in enumerate(sorted_scores) if s == score]

    avg_rank = sum(ranks) / len(ranks)
    percentile = (n - avg_rank) / (n - 1)

    return {"percentile": percentile, "rank": avg_rank}


def get_leaderboard(competition):
    """Get the leaderboard DataFrame for a competition."""
    return pd.read_csv(competition.leaderboard)


def get_medal_from_percentile(task_name: str, percentile: float, mle_bench_data_dir: str = "/checkpoint/maui_sft/winnieyangwn/datasets") -> str:
    """
    Determine the medal earned for a given percentile score.

    Args:
        task_name: Name of the competition/task
        percentile: The percentile score achieved
        mle_bench_data_dir: Path to MLE bench data directory

    Returns:
        "gold", "silver", "bronze", or "" (empty string if no medal)
    """
    if not MLEBENCH_AVAILABLE:
        return ""

    if percentile is None or (isinstance(percentile, float) and np.isnan(percentile)):
        return ""

    try:
        new_registry = registry.set_data_dir(Path(mle_bench_data_dir))
        competition = new_registry.get_competition(task_name)
        competition_leaderboard = get_leaderboard(competition)
        rank_info = competition.grader.rank_score(0, competition_leaderboard)
        is_lower_better = competition.grader.is_lower_better(competition_leaderboard)

        # Get medal thresholds as percentiles
        gold_threshold = get_rank_and_percentile(rank_info["gold_threshold"], competition_leaderboard, is_lower_better)["percentile"]
        silver_threshold = get_rank_and_percentile(rank_info["silver_threshold"], competition_leaderboard, is_lower_better)["percentile"]
        bronze_threshold = get_rank_and_percentile(rank_info["bronze_threshold"], competition_leaderboard, is_lower_better)["percentile"]

        # Determine medal
        if percentile >= gold_threshold:
            return "gold"
        elif percentile >= silver_threshold:
            return "silver"
        elif percentile >= bronze_threshold:
            return "bronze"
        else:
            return ""

    except Exception as e:
        print(f"Error getting medal for task {task_name}: {e}")
        return ""


def get_medal_thresholds(task_name: str, mle_bench_data_dir: str = "/checkpoint/maui_sft/winnieyangwn/datasets") -> dict[str, float]:
    """
    Get the medal threshold percentiles for a given task.

    Args:
        task_name: Name of the competition/task
        mle_bench_data_dir: Path to MLE bench data directory

    Returns:
        dict with "gold_threshold", "silver_threshold", "bronze_threshold" percentiles
        Returns empty dict if thresholds cannot be computed
    """
    if not MLEBENCH_AVAILABLE:
        return {}

    try:
        new_registry = registry.set_data_dir(Path(mle_bench_data_dir))
        competition = new_registry.get_competition(task_name)
        competition_leaderboard = get_leaderboard(competition)
        rank_info = competition.grader.rank_score(0, competition_leaderboard)
        is_lower_better = competition.grader.is_lower_better(competition_leaderboard)

        gold_threshold = get_rank_and_percentile(rank_info["gold_threshold"], competition_leaderboard, is_lower_better)["percentile"]
        silver_threshold = get_rank_and_percentile(rank_info["silver_threshold"], competition_leaderboard, is_lower_better)["percentile"]
        bronze_threshold = get_rank_and_percentile(rank_info["bronze_threshold"], competition_leaderboard, is_lower_better)["percentile"]

        return {
            "gold_threshold": gold_threshold,
            "silver_threshold": silver_threshold,
            "bronze_threshold": bronze_threshold,
        }

    except Exception as e:
        print(f"Error getting medal thresholds for task {task_name}: {e}")
        return {}


# ======================== PASS@K UTILITIES ========================


def _compute_normalized_mu(N: int, K: int, idx: int) -> float:
    """
    Computes the normalized weight mu_i / (N choose K) for the element at 'idx'.

    Used internally for continuous pass@k estimation.

    Args:
        N: Total batch size.
        K: Target pass@k size.
        idx: The 0-based index of the element in the sorted list.

    Returns:
        The probability that this element is the max of a random subset of size K.
    """
    if idx < K - 1:
        return 0.0

    j_values = np.arange(1, K)
    numerator = idx - j_values + 1
    denominator = N - j_values + 1
    product_term = np.prod(numerator / denominator)
    leading_factor = K / (N - K + 1)

    return leading_factor * product_term


def compute_max_g_at_k(rewards: np.ndarray, K: int) -> float:
    """
    Compute the continuous pass@k estimator (expected max score from K samples).

    This uses the unbiased estimator for the expected maximum of K samples
    drawn without replacement from N total samples.

    Args:
        rewards: numpy array of scores/rewards
        K: target k value

    Returns:
        Estimated expected maximum score at k
    """
    N = len(rewards)
    if N < K:
        return np.nan

    g_sorted = np.sort(rewards)
    weights = np.array([_compute_normalized_mu(N, K, idx) for idx in range(N)])
    return np.sum(weights * g_sorted)


def binary_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute the unbiased pass@k estimator for binary outcomes.

    Formula: 1 - (n-c choose k) / (n choose k)

    Args:
        n: Total number of samples
        c: Number of correct/passing samples
        k: Target k value

    Returns:
        Probability of at least one success in k samples
    """
    if (n - c) < k:
        return 1.0

    prob_all_fail = 1.0
    for j in range(k):
        prob_all_fail *= (n - c - j) / (n - j)
    return 1.0 - prob_all_fail


def compute_pass_at_k(df: pd.DataFrame, Ks: list[int] = None, metric: str = "percentile") -> pd.DataFrame:
    """
    Compute pass@k estimation for all tasks in the DataFrame.

    Works with both raw trajectories and processed df (from process_df).

    Args:
        df: DataFrame with either:
            - Raw: 'task' (nested dict), 'outcomes', 'metrics' columns
            - Processed: 'task_name', 'percentile', 'valid_submission' columns
        Ks: List of K values to compute
        metric: Which metric to use:
            - "percentile": Use percentile score (continuous pass@k)
            - "valid_submission": Use valid_submission flag (binary pass@k)

    Returns:
        DataFrame with tasks as index and pass@k columns
    """
    if Ks is None:
        Ks = [1, 4, 8, 16, 32, 64]
    df_work = df.copy()

    # Determine task column - use task_name if available (processed df), otherwise extract from task
    if "task_name" in df_work.columns:
        task_col = "task_name"
    elif "task" in df_work.columns:
        first_task = df_work["task"].iloc[0]
        if isinstance(first_task, dict):
            df_work["task_name"] = df_work["task"].apply(lambda x: x.get("instance_id", str(x)) if isinstance(x, dict) else str(x))
        else:
            df_work["task_name"] = df_work["task"]
        task_col = "task_name"
    else:
        task_col = None

    # Extract metric from nested structures only if not already present
    if metric == "percentile" and metric not in df_work.columns:
        if "metrics" in df_work.columns:
            df_work["percentile"] = df_work["metrics"].apply(lambda x: x.get("percentile") if isinstance(x, dict) else None)

    if metric == "valid_submission" and metric not in df_work.columns:
        if "outcomes" in df_work.columns:
            df_work["valid_submission"] = df_work["outcomes"].apply(lambda x: x.get("valid_submission", False) if isinstance(x, dict) else False)

    # Group by task
    if task_col and task_col in df_work.columns:
        grouped = df_work.groupby(task_col)[metric].apply(list)
    else:
        # If no task column, treat all as single task
        grouped = pd.Series({"all": df_work[metric].tolist()})

    # Compute pass@k for each task and each K
    all_results = {K: {} for K in Ks}

    for K in Ks:
        for task_name in grouped.index:
            values = np.array(grouped[task_name])
            values = values[~pd.isna(values)]  # Remove NaN

            if len(values) >= K:
                if metric == "valid_submission":
                    # Binary pass@k
                    n = len(values)
                    c = int(np.sum(values))
                    estimate = binary_pass_at_k(n, c, K)
                else:
                    # Continuous pass@k
                    estimate = compute_max_g_at_k(values, K)
                all_results[K][task_name] = estimate
            else:
                all_results[K][task_name] = np.nan

    # Create results DataFrame
    results_df = pd.DataFrame()
    for K in Ks:
        results_df[f"pass@{K}"] = pd.Series(all_results[K])

    return results_df


def plot_pass_at_k_score(
    results_df: pd.DataFrame,
    results_df_post: pd.DataFrame = None,
    Ks: list[int] = None,
    metric: str = "percentile",
    task_names: list[str] = None,
    save_path: str = None,
    colormap: str = "YlGnBu",
):
    """
    Plot pass@k results with each task as a separate subplot in a grid.

    Matches the style from amaia-collab pass_at_k_utils.py exactly.

    Args:
        results_df: DataFrame with pass@k results for pre-training data (tasks as index, pass@k columns)
        results_df_post: Optional DataFrame with pass@k results for post-training data
        Ks: List of K values used in the analysis
        metric: "percentile" or "valid_submission" (for ylabel)
        task_names: Optional list of task names to plot. If None, plots all tasks.
        save_path: Path to save the plot. If None, displays plot.
        colormap: Matplotlib colormap name (default: "YlGnBu")
    """
    if Ks is None:
        Ks = [1, 4, 8, 16, 32, 64]
    # Get all unique task names
    all_tasks = results_df.index.tolist()

    # Filter to specified task names if provided
    if task_names is not None:
        all_tasks = [t for t in task_names if t in results_df.index]

    n_tasks = len(all_tasks)

    if n_tasks == 0:
        print("No tasks to plot!")
        return

    # Colors from colormap (same approach as plot_metric_comparison)
    cmap = plt.get_cmap(colormap)
    # Sample colors avoiding very light colors (start from 0.3)
    pre_color = cmap(0.3 + 0.7 * 0 / max(1, 1))  # 0.3 - lighter green
    post_color = cmap(0.3 + 0.7 * 1 / max(1, 1))  # 1.0 - darker blue

    # Calculate grid size for subplots (4 columns like amaia-collab)
    n_cols = 4
    n_rows = (n_tasks + n_cols - 1) // n_cols

    # Use square subplots (4x4 each)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, task_name in enumerate(all_tasks):
        ax = axes[i]

        # Extract pre values for this task across all K values
        pre_values = [results_df.loc[task_name, f"pass@{K}"] if task_name in results_df.index else np.nan for K in Ks]

        ax.plot(Ks, pre_values, "o-", label="Pre", color=pre_color, markersize=4)

        # Only plot post if provided
        if results_df_post is not None:
            post_values = [results_df_post.loc[task_name, f"pass@{K}"] if task_name in results_df_post.index else np.nan for K in Ks]
            ax.plot(Ks, post_values, "s-", label="Post", color=post_color, markersize=4)

        # Color background based on split if available
        split = None
        if "split" in results_df.columns and task_name in results_df.index:
            split = results_df.loc[task_name, "split"]
        elif results_df_post is not None and "split" in results_df_post.columns and task_name in results_df_post.index:
            split = results_df_post.loc[task_name, "split"]
        if split == "train":
            ax.set_facecolor("#f3e8ff")  # lighter, beautiful purple
        elif split == "test":
            ax.set_facecolor("#e6f9ec")  # lighter, beautiful green

        ax.set_xlabel("K")
        ylabel = "Normalized Score" if metric == "percentile" else "Valid Submission Rate"
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(task_name, fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_average_pass_at_k_score(results_df: pd.DataFrame, results_df_post: pd.DataFrame = None, Ks: list[int] = None, metric: str = "percentile", save_path: str = None, colormap: str = "YlGnBu"):
    """
    Plot average pass@k results across all tasks.

    Matches the style from amaia-collab pass_at_k_utils.py exactly.

    Args:
        results_df: DataFrame with pass@k results for pre-training data (tasks as index, pass@k columns)
        results_df_post: Optional DataFrame with pass@k results for post-training data
        Ks: List of K values used in the analysis
        metric: "percentile" or "valid_submission" (for ylabel)
        save_path: Path to save the plot. If None, displays plot.
        colormap: Matplotlib colormap name (default: "YlGnBu")
    """
    if Ks is None:
        Ks = [1, 4, 8, 16, 32, 64]
    # Compute average pass@k across all tasks for each K from pre results
    avg_pre = [results_df[f"pass@{K}"].mean() for K in Ks]

    # Colors from colormap (same approach as plot_metric_comparison)
    cmap = plt.get_cmap(colormap)
    # Sample colors avoiding very light colors (start from 0.3)
    pre_color = cmap(0.3 + 0.7 * 0 / max(1, 1))  # 0.3 - lighter green
    post_color = cmap(0.3 + 0.7 * 1 / max(1, 1))  # 1.0 - darker blue

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(Ks, avg_pre, "o-", label="Pre", color=pre_color, markersize=6, linewidth=1.5)

    # Only compute and plot post if provided
    if results_df_post is not None:
        avg_post = [results_df_post[f"pass@{K}"].mean() for K in Ks]
        ax.plot(Ks, avg_post, "s-", label="Post", color=post_color, markersize=6, linewidth=1.5)

    ax.set_xlabel("K", fontsize=10)
    if metric == "percentile":
        ax.set_ylabel("Average Normalized Score @ pass k", fontsize=10)
    else:
        ax.set_ylabel("Average Valid Submission @ pass k", fontsize=10)
    ax.set_title("Average Pass@K Score Across All Tasks", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()

    # Print the values
    if results_df_post is not None:
        print("K\tPre\tPost\tImprovement")
        for i, K in enumerate(Ks):
            print(f"{K}\t{avg_pre[i]:.4f}\t{avg_post[i]:.4f}\t{avg_post[i] - avg_pre[i]:.4f}")
    else:
        print("K\tPre")
        for i, K in enumerate(Ks):
            print(f"{K}\t{avg_pre[i]:.4f}")


def plot_pass_at_k_valid_submission(
    results_df: pd.DataFrame, results_df_post: pd.DataFrame = None, Ks: list[int] = None, task_names: list[str] = None, save_path: str = None, colormap: str = "YlGnBu"
):
    """
    Plot pass@k results for valid submissions for pre and optionally post training data.

    Matches the style from amaia-collab pass_at_k_utils.py exactly.

    Args:
        results_df: DataFrame with pass@k valid submission results for pre-training data
        results_df_post: Optional DataFrame with pass@k valid submission results for post-training data
        Ks: List of K values used in the analysis
        task_names: Optional list of task names to plot. If None, plots all tasks.
        save_path: Path to save the plot. If None, displays plot.
        colormap: Matplotlib colormap name (default: "YlGnBu")
    """
    if Ks is None:
        Ks = [1, 4, 8, 16, 32, 64]
    # Get all unique task names from pre (or post if pre is empty)
    all_tasks = results_df.index.tolist() if len(results_df) > 0 else (results_df_post.index.tolist() if results_df_post is not None else [])

    # Filter to specified task names if provided
    if task_names is not None:
        all_tasks = [t for t in task_names if t in all_tasks]

    n_tasks = len(all_tasks)

    if n_tasks == 0:
        print("No tasks to plot!")
        return

    # Colors from colormap (same approach as plot_metric_comparison)
    cmap = plt.get_cmap(colormap)
    # Sample colors avoiding very light colors (start from 0.3)
    pre_color = cmap(0.3 + 0.7 * 0 / max(1, 1))  # 0.3 - lighter green
    post_color = cmap(0.3 + 0.7 * 1 / max(1, 1))  # 1.0 - darker blue

    # Calculate grid size for subplots
    n_cols = 4
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, task_name in enumerate(all_tasks):
        ax = axes[i]

        # Extract pre values for this task across all K values
        pre_values = [results_df.loc[task_name, f"pass@{K}"] if task_name in results_df.index else np.nan for K in Ks]

        ax.plot(Ks, pre_values, "o-", label="Pre", color=pre_color, markersize=4)

        # Only plot post if provided
        if results_df_post is not None:
            post_values = [results_df_post.loc[task_name, f"pass@{K}"] if task_name in results_df_post.index else np.nan for K in Ks]
            ax.plot(Ks, post_values, "s-", label="Post", color=post_color, markersize=4)

        # Color background based on split if available
        split = None
        if "split" in results_df.columns and task_name in results_df.index:
            split = results_df.loc[task_name, "split"]
        elif results_df_post is not None and "split" in results_df_post.columns and task_name in results_df_post.index:
            split = results_df_post.loc[task_name, "split"]
        if split == "train":
            ax.set_facecolor("#f3e8ff")  # lighter, beautiful purple
        elif split == "test":
            ax.set_facecolor("#e6f9ec")  # lighter, beautiful green

        ax.set_xlabel("K")
        ax.set_ylabel("Valid Submission @ pass k")
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(task_name, fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_pass_at_k(df: pd.DataFrame, Ks: list[int] = None, metric: str = "percentile", task_names: list[str] = None, save_path: str = None, colormap: str = "YlGnBu"):
    """
    Compute and plot pass@k curves: per-task subplots + average plot.

    Works with:
    - Raw trajectories df: 'task' (nested dict), 'outcomes', 'metrics' columns
    - Processed rollout-level df (from process_df): 'task_name', 'percentile', 'valid_submission' columns
    - Task-level df_task (from process_df): already has 'pass@{k}' and 'valid_pass@{k}' columns

    This is a convenience function that:
    1. Uses pre-computed pass@k if available (df_task), otherwise computes via compute_pass_at_k()
    2. Plots each task as a separate subplot (plot_pass_at_k_score)
    3. Plots the average across tasks (plot_average_pass_at_k_score)

    Args:
        df: DataFrame with either:
            - Raw: 'task' (nested dict), 'outcomes', 'metrics' columns
            - Processed rollout-level: 'task_name', 'percentile', 'valid_submission' columns
            - Task-level (df_task): already has 'pass@{k}' columns (tasks as index)
        Ks: List of K values to compute and plot
        metric: "percentile" for scores or "valid_submission" for binary
        task_names: Optional list of task names to plot
        save_path: Base path to save plots. If provided, saves as {path}_tasks.png and {path}_avg.png
        colormap: Matplotlib colormap name (default: "YlGnBu")

    Returns:
        results_df: DataFrame with pass@k values per task
    """
    if Ks is None:
        Ks = [1, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    # Check if df already has pass@k columns (i.e., it's df_task from process_df)
    # For valid_submission metric, look for valid_pass@k columns
    if metric == "valid_submission":
        pass_col_prefix = "valid_pass@"
    else:
        pass_col_prefix = "pass@"

    has_pass_at_k = any(col.startswith(pass_col_prefix) for col in df.columns)

    if has_pass_at_k:
        # df is already task-level with pass@k columns - use directly
        results_df = df.copy()
        # Ensure we have the right columns for the requested Ks
        expected_cols = [f"{pass_col_prefix}{K}" for K in Ks]
        available_cols = [col for col in expected_cols if col in results_df.columns]

        if not available_cols:
            print(f"Warning: No {pass_col_prefix}* columns found for requested Ks. Computing from scratch.")
            results_df = compute_pass_at_k(df, Ks=Ks, metric=metric)
        else:
            # Filter to just the pass@k columns, renaming valid_pass@k to pass@k for plotting
            results_df = results_df[available_cols].copy()
            if metric == "valid_submission":
                results_df.columns = [col.replace("valid_pass@", "pass@") for col in results_df.columns]
            # Update Ks to only include available ones
            Ks = [K for K in Ks if f"{pass_col_prefix}{K}" in df.columns]
            print(f"Using pre-computed pass@k for {len(results_df)} tasks")
    else:
        # Compute pass@k from rollout-level data
        results_df = compute_pass_at_k(df, Ks=Ks, metric=metric)
        print(f"Computed pass@k for {len(results_df)} tasks")

    print(f"Tasks: {results_df.index.tolist()}")

    # Plot per-task subplots
    task_save_path = f"{save_path}_tasks.png" if save_path else None
    plot_pass_at_k_score(results_df, Ks=Ks, metric=metric, task_names=task_names, save_path=task_save_path, colormap=colormap)

    # Plot average
    avg_save_path = f"{save_path}_avg.png" if save_path else None
    plot_average_pass_at_k_score(results_df, Ks=Ks, metric=metric, save_path=avg_save_path, colormap=colormap)

    return results_df


# ======================== METRIC COMPARISON UTILITIES ========================

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_metric_comparison(df_list: list[pd.DataFrame], labels: list[str] = None, metric: str = "percentile", agg: str = "mean", task_names: list[str] = None, colormap: str | None = "YlGnBu"):
    """
    Plot metric comparison between dataframes.

    Works with both raw trajectories and processed df (from process_df).

    Args:
        df_list: List of dataframes to compare
        labels: List of labels for each dataframe (default: ["Dataset 1", "Dataset 2", ...])
        metric: Metric to compare. Supported values:
            - "percentile": Percentile score (continuous)
            - "valid_submission": Valid submission flag (binary)
            - "any_medal", "gold_medal", "silver_medal", "bronze_medal": Medal flags (binary)
        agg: Aggregation method - "mean" or "median" (default: "mean")
        task_names: Optional list of task names to filter on. If None, use all tasks.
        colormap: Matplotlib colormap name (default: "YlGnBu")
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None, None

    # Process each DataFrame to extract the metric
    processed_dfs = []
    for df in df_list:
        df_work = df.copy()

        # Extract task_name from nested task dict only if not already present
        if "task_name" not in df_work.columns and "task" in df_work.columns:
            first_task = df_work["task"].iloc[0]
            if isinstance(first_task, dict):
                df_work["task_name"] = df_work["task"].apply(lambda x: x.get("instance_id", str(x)) if isinstance(x, dict) else str(x))
            else:
                df_work["task_name"] = df_work["task"]

        # Extract metric from nested structures only if not already present
        if metric not in df_work.columns:
            if metric == "percentile" and "metrics" in df_work.columns:
                df_work["percentile"] = df_work["metrics"].apply(lambda x: x.get("percentile") if isinstance(x, dict) else None)
            elif metric == "valid_submission" and "outcomes" in df_work.columns:
                df_work["valid_submission"] = df_work["outcomes"].apply(lambda x: x.get("valid_submission", False) if isinstance(x, dict) else False)

        # Filter by task names if provided
        if task_names is not None and "task_name" in df_work.columns:
            df_work = df_work[df_work["task_name"].isin(task_names)]

        processed_dfs.append(df_work)

    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(len(df_list))]

    # Calculate metric stats
    def calculate_metric_stats(df, metric, agg):
        if len(df) == 0 or metric not in df.columns:
            return np.nan, np.nan
        values = df[metric].dropna()
        if len(values) == 0:
            return np.nan, np.nan
        if agg == "mean":
            return values.mean(), values.std()
        else:  # median
            return values.median(), values.std()

    # Special handling for metric="medal" - create 1x4 subplots
    if metric == "medal":
        from plotly.subplots import make_subplots

        medal_types = ["any_medal", "gold_medal", "silver_medal", "bronze_medal"]
        medal_titles = ["Any Medal", "Gold Medal", "Silver Medal", "Bronze Medal"]
        medal_colors = ["#3498db", "#f1c40f", "#95a5a6", "#cd7f32"]  # Blue, Gold, Silver, Bronze

        fig = make_subplots(rows=1, cols=4, subplot_titles=medal_titles)

        all_results = {}
        for i, (medal_metric, medal_title, medal_color) in enumerate(zip(medal_types, medal_titles, medal_colors, strict=False)):
            col = i + 1

            values = []
            stds = []
            for df in processed_dfs:
                val, std = calculate_metric_stats(df, medal_metric, agg)
                values.append(val)
                stds.append(std)

            all_results[medal_metric] = {"values": values, "stds": stds}

            # Print results
            for label, val, std in zip(labels, values, stds, strict=False):
                print(f"{agg.capitalize()} {medal_title} {label}: {val:.4f} ± {std:.4f}")

            text_format = [f"{v:.2%}" if not np.isnan(v) else "N/A" for v in values]

            fig.add_trace(
                go.Bar(x=labels, y=values, text=text_format, textposition="auto", marker_color=medal_color, error_y=dict(type="data", array=stds, visible=True), name=medal_title, showlegend=False),
                row=1,
                col=col,
            )

            fig.update_yaxes(tickformat=".0%", range=[0, 1], row=1, col=col)

        fig.update_layout(title=f"{agg.capitalize()} Medal Rates Comparison", height=350, width=1000)

        fig.show()
        return all_results, None

    # Standard single-metric handling
    values = []
    stds = []
    for df in processed_dfs:
        val, std = calculate_metric_stats(df, metric, agg)
        values.append(val)
        stds.append(std)

    # Print results
    for label, val, std in zip(labels, values, stds, strict=False):
        print(f"{agg.capitalize()} {metric.capitalize()} {label}: {val:.4f} ± {std:.4f}")

    # Colors from colormap
    cmap = plt.get_cmap(colormap)
    n_items = len(df_list)
    # Sample colors from colormap, avoiding very light colors (start from 0.3)
    colors = [f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})" for c in [cmap(0.3 + 0.7 * i / max(n_items - 1, 1)) for i in range(n_items)]]

    # Format based on metric type
    is_percentage = metric in ["percentile", "any_medal", "gold_medal", "silver_medal", "bronze_medal", "valid_submission"]
    text_format = [f"{v:.2%}" for v in values] if is_percentage else [f"{v:.2f}" for v in values]

    fig = go.Figure(data=[go.Bar(x=labels, y=values, text=text_format, textposition="auto", marker_color=colors, error_y=dict(type="data", array=stds, visible=True))])

    fig.update_layout(
        title=f"{agg.capitalize()} {metric.capitalize()} Comparison",
        xaxis_title="Condition",
        yaxis_title=f"{agg.capitalize()} {metric.capitalize()}",
        yaxis_tickformat=".0%" if is_percentage else None,
        height=400,
        width=500,
    )

    fig.show()
    return values, stds


# ======================== DATA PROCESSING UTILITIES ========================


def process_df(df: pd.DataFrame, Ks: list[int] = None, mle_bench_data_dir: str = "/checkpoint/maui_sft/winnieyangwn/datasets") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a trajectories DataFrame to add medal fields and compute task-level statistics.

    This function:
    1. Extracts task_name and percentile from nested dicts
    2. Adds medal boolean fields to each rollout (any_medal, gold_medal, silver_medal, bronze_medal)
    3. Computes pass@k for each task at specified k values
    4. Computes medal rates for each task

    Args:
        df: DataFrame loaded from trajectories.jsonl with 'task', 'metrics', 'outcomes' columns
        Ks: List of K values to compute pass@k for (default: [1, 4, 8, 16, 32, 64])
        mle_bench_data_dir: Path to MLE bench data directory for medal thresholds

    Returns:
        tuple: (df, df_task) where:
            - df: Original DataFrame with added columns:
                - task_name: Extracted task identifier
                - percentile: Extracted percentile score
                - medal: Medal earned ("gold", "silver", "bronze", or "")
                - any_medal: Boolean - True if any medal earned
                - gold_medal: Boolean - True if gold medal
                - silver_medal: Boolean - True if silver medal
                - bronze_medal: Boolean - True if bronze medal
            - df_task: Task-level DataFrame indexed by task_name with columns:
                - n_rollouts: Number of rollouts for this task
                - pass@{k}: Pass@k score for each k in Ks (percentile-based)
                - valid_pass@{k}: Pass@k for valid submissions (binary)
                - any_medal_rate: Fraction of rollouts earning any medal
                - gold_medal_rate: Fraction of rollouts earning gold
                - silver_medal_rate: Fraction of rollouts earning silver
                - bronze_medal_rate: Fraction of rollouts earning bronze
    """
    if Ks is None:
        Ks = [1, 4, 8, 16, 32, 64]
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Extract task_name from nested task dict if needed
    if "task" in df.columns:
        first_task = df["task"].iloc[0]
        if isinstance(first_task, dict):
            df["task_name"] = df["task"].apply(lambda x: x.get("instance_id", str(x)) if isinstance(x, dict) else str(x))
        else:
            df["task_name"] = df["task"]

    # Extract percentile from nested metrics dict if needed
    if "metrics" in df.columns:
        df["percentile"] = df["metrics"].apply(lambda x: x.get("percentile") if isinstance(x, dict) else None)

    # Extract valid_submission from nested outcomes dict if needed
    if "outcomes" in df.columns and "valid_submission" not in df.columns:
        df["valid_submission"] = df["outcomes"].apply(lambda x: x.get("valid_submission", False) if isinstance(x, dict) else False)

    # Add medal column using get_medal_from_percentile
    df["medal"] = df.apply(lambda row: get_medal_from_percentile(row["task_name"], row["percentile"], mle_bench_data_dir), axis=1)

    # Add boolean medal fields
    df["any_medal"] = df["medal"] != ""
    df["gold_medal"] = df["medal"] == "gold"
    df["silver_medal"] = df["medal"] == "silver"
    df["bronze_medal"] = df["medal"] == "bronze"

    # ======================== Build df_task ========================

    # Get unique tasks
    task_names = df["task_name"].unique()

    # Compute pass@k for percentile scores
    pass_at_k_percentile = compute_pass_at_k(df, Ks=Ks, metric="percentile")

    # Compute pass@k for valid submissions (binary)
    pass_at_k_valid = compute_pass_at_k(df, Ks=Ks, metric="valid_submission")
    # Rename columns to distinguish from percentile pass@k
    pass_at_k_valid.columns = [f"valid_{col}" for col in pass_at_k_valid.columns]

    # Compute medal rates per task
    medal_rates = df.groupby("task_name").agg(
        n_rollouts=("task_name", "count"),
        any_medal_rate=("any_medal", "mean"),
        gold_medal_rate=("gold_medal", "mean"),
        silver_medal_rate=("silver_medal", "mean"),
        bronze_medal_rate=("bronze_medal", "mean"),
    )

    # Get medal thresholds for each task
    threshold_data = []
    for task_name in task_names:
        thresholds = get_medal_thresholds(task_name, mle_bench_data_dir)
        threshold_data.append(
            {
                "task_name": task_name,
                "gold_threshold": thresholds.get("gold_threshold", np.nan),
                "silver_threshold": thresholds.get("silver_threshold", np.nan),
                "bronze_threshold": thresholds.get("bronze_threshold", np.nan),
            }
        )
    threshold_df = pd.DataFrame(threshold_data).set_index("task_name")

    # Combine all task-level stats
    df_task = medal_rates.join(pass_at_k_percentile).join(pass_at_k_valid).join(threshold_df)

    # Reorder columns for clarity
    base_cols = ["n_rollouts"]
    pass_k_cols = [f"pass@{k}" for k in Ks]
    valid_pass_k_cols = [f"valid_pass@{k}" for k in Ks]
    medal_rate_cols = ["any_medal_rate", "gold_medal_rate", "silver_medal_rate", "bronze_medal_rate"]
    threshold_cols = ["gold_threshold", "silver_threshold", "bronze_threshold"]

    # Build final column order (only include columns that exist)
    final_cols = []
    for col in base_cols + pass_k_cols + valid_pass_k_cols + medal_rate_cols + threshold_cols:
        if col in df_task.columns:
            final_cols.append(col)
    # Add any remaining columns
    for col in df_task.columns:
        if col not in final_cols:
            final_cols.append(col)

    df_task = df_task[final_cols]

    return df, df_task


# ======================== METRIC DISTRIBUTION PLOTTING ========================


def plot_metric_distribution(
    dfs,
    task_name=None,
    labels=None,
    nbins=10,
    height=350,
    width=1400,
    valid_only=False,
    colormap="YlGnBu",
):
    """
    Plot submission analysis with 9 panels in a 3-row layout.
    All panels show side-by-side comparison of multiple datasets.

    Note: Percentile distribution and mean percentile are calculated across ALL rollouts,
    with invalid submissions treated as 0 percentile (unless valid_only=True).

    Layout:
        Row 1 (3 panels): (1) Valid Rate, (2) Percentile Distribution (all rollouts), (3) Mean Percentile Score (all rollouts)
        Row 2 (4 panels): (4) Any Medal Rate, (5) Gold Medal Rate, (6) Silver Medal Rate, (7) Bronze Medal Rate
        Row 3 (2 panels): (8) Rollout Duration Distribution, (9) Total Tokens Distribution

    Args:
        dfs: DataFrame or list of DataFrames from process_df() containing
             'task_name', 'valid_submission', 'percentile', and 'medal' columns.
             Can also work with raw trajectories DataFrames (will extract fields automatically).
        task_name: Name of the task to analyze, or list of task names (creates separate plot for each).
                   If None, analyzes all data combined.
        labels: List of labels for each dataset (e.g., ['Before', 'After RLM'])
        nbins: Number of bins for histogram (default: 10)
        height: Plot height per row in pixels (default: 350)
        width: Plot width in pixels (default: 1400)
        valid_only: If True, only include valid submissions in the analysis (default: False)
        colormap: Matplotlib colormap name (default: "YlGnBu")

    Returns:
        list of dicts: Summary statistics for each dataset including valid_count, invalid_count, all_percentiles, and medal_rates
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # If task_name is a list, loop through each task and create separate plots
    if isinstance(task_name, list):
        all_results = []
        for single_task in task_name:
            results = plot_metric_distribution(
                dfs,
                task_name=single_task,
                labels=labels,
                nbins=nbins,
                height=height,
                width=width,
                valid_only=valid_only,
                colormap=colormap,
            )
            all_results.append({"task": single_task, "results": results})
        return all_results

    # Convert single inputs to lists for uniform processing
    if not isinstance(dfs, list):
        dfs = [dfs]
    if labels is None:
        # Default to 'Before', 'After' for backward compatibility
        default_labels = ["Before", "After"] + [f"Dataset {i + 3}" for i in range(len(dfs) - 2)]
        labels = default_labels[: len(dfs)]

    # Define colors for each dataset from colormap
    cmap = plt.get_cmap(colormap)
    n_items = len(dfs)
    # Sample colors from colormap, avoiding very light colors (start from 0.3)
    colors = [f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})" for c in [cmap(0.3 + 0.7 * i / max(n_items - 1, 1)) for i in range(n_items)]]

    # Create subplot titles (note: colspan cells still need placeholder titles)
    subplot_titles = [
        "Valid Submission Rate",  # row 1, col 1
        "Percentile Distribution",  # row 1, col 2 (spans to col 3)
        "Percentile",  # row 1, col 4
        "Any Medal Rate",  # row 2, col 1
        "Gold Medal Rate",  # row 2, col 2
        "Silver Medal Rate",  # row 2, col 3
        "Bronze Medal Rate",  # row 2, col 4
        "Rollout Duration Distribution",  # row 3, col 1 (spans to col 2)
        "Total Tokens Distribution",  # row 3, col 3 (spans to col 4)
    ]

    # Create 3x4 subplots (percentile distribution spans 2 columns, row 3 has 2 histograms spanning 2 cols each)
    fig = make_subplots(
        rows=3,
        cols=4,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
        specs=[
            [{"type": "bar"}, {"type": "histogram", "colspan": 2}, None, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "histogram", "colspan": 2}, None, {"type": "histogram", "colspan": 2}, None],
        ],
    )

    results = []

    # Process each dataset
    for idx, (df, label) in enumerate(zip(dfs, labels, strict=False)):
        color = colors[idx % len(colors)]
        x_label = label

        # Make a copy to avoid modifying original
        df_work = df.copy()

        # Extract fields from nested dicts if needed (for raw trajectories)
        if "task_name" not in df_work.columns and "task" in df_work.columns:
            first_task = df_work["task"].iloc[0]
            if isinstance(first_task, dict):
                df_work["task_name"] = df_work["task"].apply(lambda x: x.get("instance_id", str(x)) if isinstance(x, dict) else str(x))
            else:
                df_work["task_name"] = df_work["task"]

        if "percentile" not in df_work.columns and "metrics" in df_work.columns:
            df_work["percentile"] = df_work["metrics"].apply(lambda x: x.get("percentile") if isinstance(x, dict) else None)

        if "valid_submission" not in df_work.columns and "outcomes" in df_work.columns:
            df_work["valid_submission"] = df_work["outcomes"].apply(lambda x: x.get("valid_submission", False) if isinstance(x, dict) else False)

        # Filter to task if specified, otherwise use all data
        if task_name is not None:
            # Handle both string and list inputs for task_name
            task_names_list = task_name if isinstance(task_name, list) else [task_name]
            # Try 'task_name' column first, then 'task_id'
            if "task_name" in df_work.columns:
                task_df = df_work[df_work["task_name"].isin(task_names_list)]
            elif "task_id" in df_work.columns:
                task_df = df_work[df_work["task_id"].isin(task_names_list)]
            else:
                task_df = df_work
        else:
            task_df = df_work

        # Store original counts before filtering
        original_total = len(task_df)
        original_valid = task_df["valid_submission"].sum()

        # Filter to only valid submissions if valid_only=True
        if valid_only:
            task_df = task_df[task_df["valid_submission"] == True]

        valid_count = task_df["valid_submission"].sum()
        invalid_count = len(task_df) - valid_count
        total_count = len(task_df)
        valid_rate = original_valid / original_total * 100 if original_total > 0 else 0

        # Calculate standard error for valid rate (binomial proportion) - use original counts
        valid_rate_se = np.sqrt((valid_rate / 100) * (1 - valid_rate / 100) / original_total) * 100 if original_total > 0 else 0

        # Get percentiles from the dataframe's 'percentile' column for ALL rollouts
        # Invalid submissions are treated as 0 percentile
        task_percentiles = task_df["percentile"].values
        valid_mask = task_df["valid_submission"].values
        # All percentiles: use 0 for invalid submissions, keep valid percentiles as-is
        all_percentiles = []
        for p, v in zip(task_percentiles, valid_mask, strict=False):
            if v and p is not None and not np.isnan(p):
                all_percentiles.append(p)
            else:
                all_percentiles.append(0.0)  # Invalid submissions count as 0 percentile
        mean_percentile = np.mean(all_percentiles) * 100 if all_percentiles else 0
        # Calculate standard error for mean percentile
        percentile_se = (np.std(all_percentiles) / np.sqrt(len(all_percentiles))) * 100 if len(all_percentiles) > 1 else 0

        # Panel 1: Valid Submission Rate (grouped bar)
        fig.add_trace(
            go.Bar(
                x=[x_label],
                y=[valid_rate],
                name=x_label,
                marker_color=color,
                text=[f"{valid_rate:.1f}%"],
                textposition="auto",
                error_y=dict(type="data", array=[valid_rate_se], visible=True),
                showlegend=True,
                legendgroup=x_label,
            ),
            row=1,
            col=1,
        )

        # Panel 2: Percentile Distribution (overlaid histograms) - ALL rollouts
        fig.add_trace(
            go.Histogram(
                x=all_percentiles,
                nbinsx=nbins,
                name=x_label,
                marker_color=color,
                showlegend=False,
                legendgroup=x_label,
            ),
            row=1,
            col=2,
        )

        # Panel 3: Mean Percentile Score (grouped bar) - in column 4 due to colspan
        fig.add_trace(
            go.Bar(
                x=[x_label],
                y=[mean_percentile],
                name=x_label,
                marker_color=color,
                text=[f"{mean_percentile:.1f}%"],
                textposition="auto",
                error_y=dict(type="data", array=[percentile_se], visible=True),
                showlegend=False,
                legendgroup=x_label,
            ),
            row=1,
            col=4,
        )

        # Get medal rates directly from df's medal column (or medal boolean fields)
        medal_rates = None
        medal_rates_se = {}
        if total_count > 0:
            # Check for medal boolean fields first (from process_df), then fall back to medal string column
            if "any_medal" in task_df.columns:
                gold_count = task_df["gold_medal"].sum() if "gold_medal" in task_df.columns else 0
                silver_count = task_df["silver_medal"].sum() if "silver_medal" in task_df.columns else 0
                bronze_count = task_df["bronze_medal"].sum() if "bronze_medal" in task_df.columns else 0
                any_medal_count = task_df["any_medal"].sum()
            elif "medal" in task_df.columns:
                # Calculate medal rates from the medal column
                medals = task_df["medal"].values
                gold_count = sum(1 for m in medals if m == "gold")
                silver_count = sum(1 for m in medals if m == "silver")
                bronze_count = sum(1 for m in medals if m == "bronze")
                any_medal_count = gold_count + silver_count + bronze_count
            else:
                gold_count = silver_count = bronze_count = any_medal_count = 0

            medal_rates = {
                "gold_rate": gold_count / total_count,
                "silver_rate": silver_count / total_count,
                "bronze_rate": bronze_count / total_count,
                "any_medal_rate": any_medal_count / total_count,
            }

            # Calculate standard errors for medal rates (binomial proportion)
            medal_rates_se = {
                "any_medal_se": np.sqrt(medal_rates["any_medal_rate"] * (1 - medal_rates["any_medal_rate"]) / total_count) * 100 if total_count > 0 else 0,
                "gold_se": np.sqrt(medal_rates["gold_rate"] * (1 - medal_rates["gold_rate"]) / total_count) * 100 if total_count > 0 else 0,
                "silver_se": np.sqrt(medal_rates["silver_rate"] * (1 - medal_rates["silver_rate"]) / total_count) * 100 if total_count > 0 else 0,
                "bronze_se": np.sqrt(medal_rates["bronze_rate"] * (1 - medal_rates["bronze_rate"]) / total_count) * 100 if total_count > 0 else 0,
            }

            # Panel 4: Any Medal Rate
            fig.add_trace(
                go.Bar(
                    x=[x_label],
                    y=[medal_rates["any_medal_rate"] * 100],
                    name=x_label,
                    marker_color=color,
                    text=[f"{medal_rates['any_medal_rate'] * 100:.1f}%"],
                    textposition="outside",
                    error_y=dict(type="data", array=[medal_rates_se["any_medal_se"]], visible=True),
                    showlegend=False,
                    legendgroup=x_label,
                ),
                row=2,
                col=1,
            )

            # Panel 5: Gold Medal Rate
            fig.add_trace(
                go.Bar(
                    x=[x_label],
                    y=[medal_rates["gold_rate"] * 100],
                    name=x_label,
                    marker_color=color,
                    text=[f"{medal_rates['gold_rate'] * 100:.1f}%"],
                    textposition="outside",
                    error_y=dict(type="data", array=[medal_rates_se["gold_se"]], visible=True),
                    showlegend=False,
                    legendgroup=x_label,
                ),
                row=2,
                col=2,
            )

            # Panel 6: Silver Medal Rate
            fig.add_trace(
                go.Bar(
                    x=[x_label],
                    y=[medal_rates["silver_rate"] * 100],
                    name=x_label,
                    marker_color=color,
                    text=[f"{medal_rates['silver_rate'] * 100:.1f}%"],
                    textposition="outside",
                    error_y=dict(type="data", array=[medal_rates_se["silver_se"]], visible=True),
                    showlegend=False,
                    legendgroup=x_label,
                ),
                row=2,
                col=3,
            )

            # Panel 7: Bronze Medal Rate
            fig.add_trace(
                go.Bar(
                    x=[x_label],
                    y=[medal_rates["bronze_rate"] * 100],
                    name=x_label,
                    marker_color=color,
                    text=[f"{medal_rates['bronze_rate'] * 100:.1f}%"],
                    textposition="outside",
                    error_y=dict(type="data", array=[medal_rates_se["bronze_se"]], visible=True),
                    showlegend=False,
                    legendgroup=x_label,
                ),
                row=2,
                col=4,
            )

        # Extract rollout_duration - try direct column first, then nested metrics
        # Convert from seconds to minutes
        rollout_durations = []
        if "rollout_duration" in task_df.columns:
            rollout_durations = [d / 60.0 for d in task_df["rollout_duration"].dropna().tolist()]
        elif "metrics" in task_df.columns:
            for m in task_df["metrics"]:
                if isinstance(m, dict) and "rollout_duration" in m:
                    val = m.get("rollout_duration")
                    if val is not None and not np.isnan(val):
                        rollout_durations.append(val / 60.0)  # Convert seconds to minutes

        # Extract total_tokens - try direct column first, then nested metrics
        total_tokens = []
        if "total_tokens" in task_df.columns:
            total_tokens = task_df["total_tokens"].dropna().tolist()
        elif "metrics" in task_df.columns:
            for m in task_df["metrics"]:
                if isinstance(m, dict) and "total_tokens" in m:
                    val = m.get("total_tokens")
                    if val is not None and not np.isnan(val):
                        total_tokens.append(val)

        # Panel 8: Rollout Duration Distribution (row 3, col 1)
        if rollout_durations:
            fig.add_trace(
                go.Histogram(
                    x=rollout_durations,
                    nbinsx=nbins,
                    name=x_label,
                    marker_color=color,
                    showlegend=False,
                    legendgroup=x_label,
                ),
                row=3,
                col=1,
            )

        # Panel 9: Total Tokens Distribution (row 3, col 3)
        if total_tokens:
            fig.add_trace(
                go.Histogram(
                    x=total_tokens,
                    nbinsx=nbins,
                    name=x_label,
                    marker_color=color,
                    showlegend=False,
                    legendgroup=x_label,
                ),
                row=3,
                col=3,
            )

        results.append(
            {
                "label": label,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "total_count": total_count,
                "valid_rate": valid_rate,
                "all_percentiles": all_percentiles,
                "mean_percentile": mean_percentile,
                "medal_rates": medal_rates,
                "rollout_durations": rollout_durations,
                "total_tokens": total_tokens,
            }
        )

    # Update axes labels - Row 1
    fig.update_yaxes(title_text="Pass@1", ticksuffix="%", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Pass@1", ticksuffix="%", row=1, col=4)

    # Update axes labels - Row 2
    fig.update_yaxes(title_text="Pass@1", ticksuffix="%", row=2, col=1)
    fig.update_yaxes(title_text="Pass@1", ticksuffix="%", row=2, col=2)
    fig.update_yaxes(title_text="Pass@1", ticksuffix="%", row=2, col=3)
    fig.update_yaxes(title_text="Pass@1", ticksuffix="%", row=2, col=4)

    # Update axes labels - Row 3
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=3)
    fig.update_xaxes(title_text="Duration (minutes)", row=3, col=1)
    fig.update_xaxes(title_text="Tokens", row=3, col=3)

    # Update x-axis for percentile distribution
    fig.update_xaxes(title_text="Percentile", tickformat=".0%", range=[-0.05, 1.05], row=1, col=2)

    # Set y-axis range for rate panels
    fig.update_yaxes(range=[0, 105], row=1, col=1)
    fig.update_yaxes(range=[0, 105], row=1, col=4)
    fig.update_yaxes(range=[0, 105], row=2, col=1)
    fig.update_yaxes(range=[0, 105], row=2, col=2)
    fig.update_yaxes(range=[0, 105], row=2, col=3)
    fig.update_yaxes(range=[0, 105], row=2, col=4)

    fig.update_layout(
        title_text=f"Submission Analysis{' (task: ' + task_name + ')' if task_name else ''}",
        height=height * 3,
        width=width,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        barmode="group",
    )

    fig.show()

    # Print statistics for each dataset
    for result in results:
        print(f"\n{'=' * 50}")
        print(f"{result['label']}{' - Task: ' + task_name if task_name else ''}")
        print(f"{'=' * 50}")
        print(f"Valid submissions: {result['valid_count']} ({result['valid_count'] / result['total_count']:.1%})")
        print(f"Invalid submissions: {result['invalid_count']} ({result['invalid_count'] / result['total_count']:.1%})")
        print(f"Total submissions: {result['total_count']}")

        if result["all_percentiles"]:
            print("\nAll rollout percentiles (invalid=0):")
            print(f"  Mean: {np.mean(result['all_percentiles']):.2%}")
            print(f"  Median: {np.median(result['all_percentiles']):.2%}")
            print(f"  Max: {np.max(result['all_percentiles']):.2%}")
            print(f"  Min: {np.min(result['all_percentiles']):.2%}")

        if result["medal_rates"]:
            print("\nMedal rates:")
            print(f"  Gold: {result['medal_rates']['gold_rate']:.1%}")
            print(f"  Silver: {result['medal_rates']['silver_rate']:.1%}")
            print(f"  Bronze: {result['medal_rates']['bronze_rate']:.1%}")
            print(f"  Any Medal: {result['medal_rates']['any_medal_rate']:.1%}")

        if result["rollout_durations"]:
            print("\nRollout duration (minutes):")
            print(f"  Mean: {np.mean(result['rollout_durations']):.1f}")
            print(f"  Median: {np.median(result['rollout_durations']):.1f}")
            print(f"  Max: {np.max(result['rollout_durations']):.1f}")
            print(f"  Min: {np.min(result['rollout_durations']):.1f}")

        if result["total_tokens"]:
            print("\nTotal tokens:")
            print(f"  Mean: {np.mean(result['total_tokens']):.0f}")
            print(f"  Median: {np.median(result['total_tokens']):.0f}")
            print(f"  Max: {np.max(result['total_tokens']):.0f}")
            print(f"  Min: {np.min(result['total_tokens']):.0f}")

    return results
