"""MLE-bench agent plugin for rLLM.

This plugin provides an agent that solves Kaggle ML competitions
using sandboxed GPU containers via AgentBox.
"""

from mle_bench_agent.prompts import (
    BASH_TOOL,
    CHECK_SUBMISSION_COMMAND,
    CHECK_SUBMISSION_TOOL,
    DATA_INFO_COMMAND,
    FORMAT_ERROR_MSG,
    INSTANCE_PROMPT,
    SUBMIT_TOOL,
    SYSTEM_PROMPT,
    truncate_output,
)

# These will be implemented in later steps
# from mle_bench_agent.agent import MLEBenchAgentFlow, mle_bench_agent
# from mle_bench_agent.evaluator import MLEBenchEvaluator

__all__ = [
    "SYSTEM_PROMPT",
    "INSTANCE_PROMPT",
    "FORMAT_ERROR_MSG",
    "BASH_TOOL",
    "SUBMIT_TOOL",
    "CHECK_SUBMISSION_TOOL",
    "DATA_INFO_COMMAND",
    "CHECK_SUBMISSION_COMMAND",
    "truncate_output",
]
