"""Prompts and tool schemas for MLE-bench agent.

Ported from amaia-collab/apps/sea/envs/envs/mle_bench/prompts/
"""

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an AUTONOMOUS ML engineering agent. Your goal is to
develop a solution that trains a model and generates predictions for a
Kaggle-style ML competition.

Your solution MUST:
1. Train a model during the evaluation run
2. Generate predictions for the test set in /root/data/
3. Write predictions to /workspace/submission.csv
4. Complete within {eval_timeout_hrs} hours on 1× H200 GPU, 120GB RAM
5. Match test set IDs perfectly

Environment:
- Each tool call starts fresh; only filesystem persists
- Individual tool calls timeout after {timeout_min} minutes
- Total context window: {context_size} tokens
- Network: Enabled for pip/mamba installs
- Data location: /root/data/ (read-only)
- Working directory: /workspace

Critical warnings:
- argparse.parse_args() crashes in non-interactive shell — use parse_args([]) or hardcode params
- Do NOT upgrade numpy to 2.x (breaks pandas)
- Always verify submission.csv format before submitting
""".strip()


# ============================================================================
# Instance Prompt
# ============================================================================

INSTANCE_PROMPT = """<task_description>
{task_description}
</task_description>

<data_info>
{data_info}
</data_info>

Solve the task following the guidelines above. When you are confident your
solution is correct, use the submit tool with the path to your solution.py.
""".strip()


# ============================================================================
# Error Messages
# ============================================================================

FORMAT_ERROR_MSG = "You must use one of the available tools (bash or submit) to interact with the environment. Please try again with a valid tool call."


# ============================================================================
# OpenAI Function-Calling Tool Schemas
# ============================================================================

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in the container.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": ("Submit your final solution. The solution.py file must write predictions to /workspace/submission.csv when executed."),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the solution script (default: /workspace/solution.py).",
                }
            },
            "required": ["path"],
        },
    },
}

CHECK_SUBMISSION_TOOL = {
    "type": "function",
    "function": {
        "name": "check_submission_validity",
        "description": ("Run solution.py and validate that the generated submission.csv matches the expected format (column names, row count, value ranges)."),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


# ============================================================================
# Helper Commands
# ============================================================================

DATA_INFO_COMMAND = """cd /root/data && \
echo "=== DATA STRUCTURE ===" && ls -sh && \
echo -e "\\n=== CSV ROW COUNTS ===" && wc -l *.csv 2>/dev/null && \
echo -e "\\n=== SAMPLE SUBMISSION FORMAT ===" && head -3 sample_submission.csv 2>/dev/null"""

CHECK_SUBMISSION_COMMAND = "cd /workspace && python solution.py && echo '=== SUBMISSION HEAD ===' && head -5 submission.csv && echo '=== SUBMISSION SHAPE ===' && wc -l submission.csv"


# ============================================================================
# Output Truncation
# ============================================================================

MAX_OUTPUT_LENGTH = 16000  # Characters


def truncate_output(output: str, max_length: int = MAX_OUTPUT_LENGTH) -> str:
    """Truncate long output to avoid context overflow."""
    if len(output) <= max_length:
        return output

    half = max_length // 2
    return output[:half] + f"\n\n... [TRUNCATED {len(output) - max_length} characters] ...\n\n" + output[-half:]
