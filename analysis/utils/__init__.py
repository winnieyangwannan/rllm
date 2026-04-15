"""Analysis utilities for rLLM evaluations."""

from .eval_utils import (
    INFRA_ERROR_CATEGORIES,
    MLEBENCH_AVAILABLE,
    analyze_invalid_submissions,
    binary_pass_at_k,
    categorize_error,
    # Pass@k utilities
    compute_max_g_at_k,
    compute_pass_at_k,
    flatten_outcomes,
    get_leaderboard,
    get_medal_from_percentile,
    # Medal utilities
    get_rank_and_percentile,
    get_summary_stats,
    is_infra_error_category,
    plot_average_pass_at_k_score,
    # Metric comparison
    plot_metric_comparison,
    plot_pass_at_k,
    plot_pass_at_k_score,
    plot_pass_at_k_valid_submission,
    plot_submission_validity_breakdown,
)

__all__ = [
    "categorize_error",
    "is_infra_error_category",
    "plot_submission_validity_breakdown",
    "analyze_invalid_submissions",
    "flatten_outcomes",
    "get_summary_stats",
    "INFRA_ERROR_CATEGORIES",
    # Medal utilities
    "get_rank_and_percentile",
    "get_leaderboard",
    "get_medal_from_percentile",
    "MLEBENCH_AVAILABLE",
    # Pass@k utilities
    "compute_max_g_at_k",
    "binary_pass_at_k",
    "compute_pass_at_k",
    "plot_pass_at_k",
    "plot_pass_at_k_score",
    "plot_average_pass_at_k_score",
    "plot_pass_at_k_valid_submission",
    # Metric comparison
    "plot_metric_comparison",
]
