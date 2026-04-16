"""MLE-bench agent plugin for rLLM.

This plugin provides an agent that solves Kaggle ML competitions
using sandboxed GPU containers via AgentBox.
"""

from mle_agent.agent import MLEAgentFlow, _run_agent_loop, mle_agent
from mle_agent.evaluator import MLEEvaluator
from mle_agent.prompts import (
    DATA_INFO_COMMAND,
    FORMAT_ERROR_MSG,
    INSTANCE_PROMPT,
    SYSTEM_PROMPT,
)
from mle_agent.tools import (
    BASH_TOOL,
    CHECK_SUBMISSION_VALIDITY_TOOL,
    CREATE_TOOL,
    EDIT_TOOL,
    MLE_BENCH_TOOLS,
    MLE_BENCH_TOOLS_BASIC,
    MLE_BENCH_TOOLS_CODE,
    MLE_BENCH_TOOLS_CSV,
    SUBMIT_TOOL,
    SUBMIT_TOOL_CODE,
    SUBMIT_TOOL_CSV,
    BashResult,
    WorkerDeadError,
    execute_tool,
    get_tools,
    truncate_output,
)

__all__ = [
    "MLEAgentFlow",
    "mle_agent",
    "_run_agent_loop",
    "MLEEvaluator",
    # Prompts
    "SYSTEM_PROMPT",
    "INSTANCE_PROMPT",
    "FORMAT_ERROR_MSG",
    "DATA_INFO_COMMAND",
    # Tools
    "BASH_TOOL",
    "BashResult",
    "EDIT_TOOL",
    "CREATE_TOOL",
    "SUBMIT_TOOL",
    "SUBMIT_TOOL_CODE",
    "SUBMIT_TOOL_CSV",
    "CHECK_SUBMISSION_VALIDITY_TOOL",
    "MLE_BENCH_TOOLS",
    "MLE_BENCH_TOOLS_BASIC",
    "MLE_BENCH_TOOLS_CODE",
    "MLE_BENCH_TOOLS_CSV",
    "WorkerDeadError",
    "execute_tool",
    "get_tools",
    "truncate_output",
]
