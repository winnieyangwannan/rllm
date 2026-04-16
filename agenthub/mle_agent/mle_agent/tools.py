"""Tool schemas and execution for MLE-bench agent.

Ported from amaia-collab/apps/sea/lib/impgen/tool_definitions.py

Tools available for MLE-bench (mle_bench_bash_env_with_csv_check):
1. bash - Execute bash commands
2. edit - Search/replace in files
3. create - Create new files
4. submit - Submit train.py + submission.csv
5. check_submission_validity - Validate submission format
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, TypedDict

from rllm.sdk.sandbox.protocol import Sandbox

logger = logging.getLogger(__name__)


# ============================================================================
# BashResult (aligned with amaia-collab SessionOutput/BashResult)
# ============================================================================


class BashResult(TypedDict):
    """Result from executing a bash command.

    Aligned with amaia-collab's SessionOutput structure.
    """

    status: Literal["success", "error"]
    output: str
    error_type: Literal["none", "timeout", "too_long", "exit", "broken_pipe", "other"]
    exit_code: int


# ============================================================================
# Tool Schemas (OpenAI function calling format)
# ============================================================================

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Executes bash command in the current session.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute, e.g. ls /root/data",
                },
            },
            "required": ["command"],
        },
    },
}

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": "Replaces [search_lines] from [path] with [replacement_lines], where [path] must exist "
        "and [search_lines] must uniquely and exactly match one or more consecutive lines from the original file, "
        "including indentations and whitespaces.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to file to edit.",
                },
                "search_lines": {
                    "type": "string",
                    "description": "Original lines to replace.",
                },
                "replacement_lines": {
                    "type": "string",
                    "description": "New lines to replace [search_lines] with.",
                },
            },
            "required": ["path", "search_lines", "replacement_lines"],
        },
    },
}

CREATE_TOOL = {
    "type": "function",
    "function": {
        "name": "create",
        "description": "Creates a new file at [path] with [content], where [path] must not exist, but its parent directory must exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to create new file at.",
                },
                "content": {
                    "type": "string",
                    "description": "New file contents to write.",
                },
            },
            "required": ["path", "content"],
        },
    },
}

# ============================================================================
# Submit Tool - CODE mode (agent submits train.py, evaluator executes it)
# ============================================================================

SUBMIT_TOOL_CODE = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submits your final training script for evaluation. The script will be executed to train the model and generate /workspace/submission.csv.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the training Python file (.py). Should be /workspace/train.py. This script will be executed to train your model and generate predictions.",
                },
            },
            "required": ["path"],
        },
    },
}

# ============================================================================
# Submit Tool - CSV mode (agent runs train.py, generates CSV, submits both)
# ============================================================================

SUBMIT_TOOL_CSV = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submits your final solution. Requires a training script and test set predictions. "
        "You must have already run the training script and generated submission.csv before calling this.",
        "parameters": {
            "type": "object",
            "properties": {
                "train_path": {
                    "type": "string",
                    "description": "Path to training Python file (.py). Should be /workspace/train.py. This is the final script that trains your model and generates predictions on test set.",
                },
                "submission_path": {
                    "type": "string",
                    "description": "Path to the test set prediction CSV file. Should be /workspace/submission.csv.",
                },
            },
            "required": ["train_path", "submission_path"],
        },
    },
}

# Backward compatibility alias
SUBMIT_TOOL = SUBMIT_TOOL_CSV

# ============================================================================
# Check Submission Validity Tool
# ============================================================================

CHECK_SUBMISSION_VALIDITY_TOOL = {
    "type": "function",
    "function": {
        "name": "check_submission_validity",
        "description": "Checks if the training script and submission CSV file are valid for the current competition. "
        "Validates that train.py exists and submission.csv has correct format (column names, row count, data types, etc.) "
        "without providing a score. Use this to validate your submission before final submit. Similar to Kaggle's public leaderboard validation.",
        "parameters": {
            "type": "object",
            "properties": {
                "train_path": {
                    "type": "string",
                    "description": "Path to the training Python file (.py). Should be /workspace/train.py.",
                },
                "submission_path": {
                    "type": "string",
                    "description": "Path to the submission CSV file to validate. Should be /workspace/submission.csv.",
                },
            },
            "required": ["train_path", "submission_path"],
        },
    },
}

# ============================================================================
# Tool Sets for Different Modes
# ============================================================================

# CSV mode: Agent runs train.py, generates submission.csv, then submits both
# (matches amaia-collab mle_bench_bash_env_with_csv_check)
MLE_BENCH_TOOLS_CSV = [
    BASH_TOOL,
    EDIT_TOOL,
    CREATE_TOOL,
    SUBMIT_TOOL_CSV,
    CHECK_SUBMISSION_VALIDITY_TOOL,
]

# CODE mode: Agent submits just train.py, evaluator executes it
MLE_BENCH_TOOLS_CODE = [
    BASH_TOOL,
    EDIT_TOOL,
    CREATE_TOOL,
    SUBMIT_TOOL_CODE,
]

# Default (CSV mode with check_submission_validity)
MLE_BENCH_TOOLS = MLE_BENCH_TOOLS_CSV

# Backward compatibility
MLE_BENCH_TOOLS_BASIC = MLE_BENCH_TOOLS_CODE


def get_tools(submit_file: str = "csv", check_submission_validity: bool = True) -> list[dict]:
    """Get the appropriate tool set based on mode.

    Args:
        submit_file: "csv" (agent generates CSV) or "code" (evaluator runs train.py)
        check_submission_validity: Whether to include validation tool

    Returns:
        List of tool schemas for OpenAI function calling
    """
    if submit_file == "code":
        if check_submission_validity:
            return MLE_BENCH_TOOLS_CODE + [CHECK_SUBMISSION_VALIDITY_TOOL]
        else:
            return MLE_BENCH_TOOLS_CODE
    else:  # csv mode
        if check_submission_validity:
            return MLE_BENCH_TOOLS_CSV
        else:
            return [BASH_TOOL, EDIT_TOOL, CREATE_TOOL, SUBMIT_TOOL_CSV]


# ============================================================================
# Tool Execution
# ============================================================================

MAX_OUTPUT_LENGTH = 16000  # Characters
# Limit to 480KB (~96k tokens) for very long outputs - matches amaia-collab
MAX_BUFFER_SIZE = 480 * 1024


def truncate_output(output: str, max_length: int = MAX_OUTPUT_LENGTH) -> str:
    """Truncate long output to avoid context overflow."""
    if len(output) <= max_length:
        return output

    half = max_length // 2
    return output[:half] + f"\n\n... [TRUNCATED {len(output) - max_length} characters] ...\n\n" + output[-half:]


class WorkerDeadError(Exception):
    """Raised when the agentbox worker hosting the container is unreachable.

    This error is unrecoverable - the container must be recreated on a new worker.
    """

    pass


def _safe_exec(
    sandbox: Sandbox,
    command: str,
    timeout: float = 60.0,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> tuple[str, int]:
    """Execute command in sandbox with timeout and retry logic.

    Returns:
        Tuple of (output, exit_code). exit_code is -1 on exception.

    Raises:
        WorkerDeadError: If the worker is permanently unreachable (connection refused).
    """
    import time

    consecutive_connection_failures = 0
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            output = sandbox.exec(command, timeout=timeout)
            # Success - reset failure counter
            return output, 0
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Detect dead worker patterns
            is_connection_error = any(
                pattern in error_str
                for pattern in [
                    "connection refused",
                    "unavailable",
                    "failed to connect",
                    "shell session failed to start",
                ]
            )

            if is_connection_error:
                consecutive_connection_failures += 1
                logger.warning("Sandbox connection error (attempt %d/%d): %s", attempt + 1, max_retries + 1, e)

                # After 3 consecutive connection failures, worker is likely dead
                if consecutive_connection_failures >= 3:
                    raise WorkerDeadError(f"Worker appears dead after {consecutive_connection_failures} consecutive connection failures: {e}") from e

                # Wait before retry
                if attempt < max_retries:
                    time.sleep(retry_delay * (attempt + 1))
            else:
                # Non-connection error - don't retry
                return f"Error: {e}", -1

    # Max retries exceeded
    return f"Error after {max_retries + 1} attempts: {last_error}", -1


def _exec(sandbox: Sandbox, command: str, timeout: float = 60.0) -> str:
    """Execute command in sandbox, returning just output string.

    Convenience wrapper around _safe_exec for simple use cases.
    """
    output, _ = _safe_exec(sandbox, command, timeout)
    return output


def _run_bash_with_result(
    sandbox: Sandbox,
    command: str,
    timeout: float = 300.0,
) -> BashResult:
    """Execute bash command and return structured BashResult.

    Aligned with amaia-collab's SessionOutput structure.
    """
    timeout_secs = int(timeout)

    # Wrap command with timeout to enforce at OS level
    # Using --signal=KILL to ensure force-kill after grace period
    # Also capture exit code
    wrapped_command = f"timeout --signal=KILL {timeout_secs} bash -c {_shell_quote(command)}; echo __EXIT_CODE__=$?"

    output, _ = _safe_exec(sandbox, wrapped_command, timeout=timeout + 30)

    # Parse exit code from output
    exit_code = 0
    if "__EXIT_CODE__=" in output:
        parts = output.rsplit("__EXIT_CODE__=", 1)
        output = parts[0]
        try:
            exit_code = int(parts[1].strip().split()[0])
        except (ValueError, IndexError):
            exit_code = -1

    # Determine status and error_type
    status: Literal["success", "error"] = "success"
    error_type: Literal["none", "timeout", "too_long", "exit", "broken_pipe", "other"] = "none"

    # Check for timeout (exit code 124 or 137 from timeout command)
    if exit_code == 124 or exit_code == 137:
        status = "error"
        error_type = "timeout"
    elif "Killed" in output[-200:]:
        status = "error"
        error_type = "timeout"
    elif exit_code != 0:
        status = "error"
        error_type = "exit"

    # Check for too_long (output exceeds buffer)
    if len(output) > MAX_BUFFER_SIZE:
        error_type = "too_long"
        output = output[:MAX_BUFFER_SIZE]

    return BashResult(
        status=status,
        output=output,
        error_type=error_type,
        exit_code=exit_code,
    )


def _format_bash_output(result: BashResult, timeout_secs: int) -> str:
    """Format BashResult for display, matching amaia-collab's _format_action_observation."""
    lines = []

    if result["status"] == "success":
        lines.append("Backend successfully executed command:")
    else:
        if result["error_type"] == "timeout":
            lines.append(f"Backend timed out (>{timeout_secs}s):")
        elif result["error_type"] == "too_long":
            lines.append("Backend output too long (truncated):")
        else:
            lines.append(f"Backend failed to execute command (exit code {result['exit_code']}):")

    if result["output"]:
        lines.append(truncate_output(result["output"]))

    return "\n".join(lines)


def execute_bash(
    sandbox: Sandbox,
    args: dict[str, Any],
    timeout: float = 300.0,
) -> str:
    """Execute bash command with OS-level timeout enforcement.

    Returns formatted output matching amaia-collab's format:
    - "Backend successfully executed command:" for success
    - "Backend timed out:" for timeout
    - "Backend failed to execute command:" for other errors
    """
    command = args.get("command", "")
    if not command:
        return "Error: No command provided"

    result = _run_bash_with_result(sandbox, command, timeout)
    return _format_bash_output(result, int(timeout))


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell usage."""
    # Use single quotes and escape any existing single quotes
    return "'" + s.replace("'", "'\"'\"'") + "'"


def execute_edit(
    sandbox: Sandbox,
    args: dict[str, Any],
    timeout: float = 60.0,
) -> str:
    """Execute edit tool - search/replace in file.

    Uses a Python script approach similar to amaia-collab plugins.
    """
    path = args.get("path", "")
    search_lines = args.get("search_lines", "")
    replacement_lines = args.get("replacement_lines", "")

    if not path:
        return "Error: No path provided"
    if not search_lines:
        return "Error: No search_lines provided"

    # Use heredoc to pass the search/replace content safely
    # Escape single quotes in content
    search_escaped = search_lines.replace("'", "'\"'\"'")
    replacement_escaped = replacement_lines.replace("'", "'\"'\"'")

    # Python script for search/replace
    script = f'''
import sys
path = "{path}"
try:
    with open(path, 'r') as f:
        content = f.read()
except FileNotFoundError:
    print(f"Error: File not found: {{path}}")
    sys.exit(1)

search = \'\'\'\\
{search_escaped}\'\'\'

replacement = \'\'\'\\
{replacement_escaped}\'\'\'

# Check if search string exists
if search not in content:
    print(f"Error: search_lines not found in {{path}}")
    print("Make sure search_lines exactly matches the file content including whitespace.")
    sys.exit(1)

# Check for unique match
count = content.count(search)
if count > 1:
    print(f"Error: search_lines matches {{count}} locations in {{path}}. Must be unique.")
    sys.exit(1)

# Perform replacement
new_content = content.replace(search, replacement, 1)

with open(path, 'w') as f:
    f.write(new_content)

print(f"Successfully edited {{path}}")
'''

    # Write script to temp file and execute
    cmd = f"python3 -c '{script}'"
    output = _exec(sandbox, cmd, timeout=timeout)
    return output


def execute_create(
    sandbox: Sandbox,
    args: dict[str, Any],
    timeout: float = 60.0,
) -> str:
    """Execute create tool - create new file."""
    path = args.get("path", "")
    content = args.get("content", "")

    if not path:
        return "Error: No path provided"

    # Check if file already exists
    # Use distinct markers to avoid substring matching issues (EXISTS vs NOT_EXISTS)
    check_cmd = f"test -f {path} && echo 'FILE_FOUND' || echo 'FILE_MISSING'"
    result = _exec(sandbox, check_cmd, timeout=10)

    if "FILE_FOUND" in result:
        return f"Error: File already exists: {path}. Use edit tool to modify existing files."

    # Check if parent directory exists
    parent = str(Path(path).parent)
    check_parent = f"test -d {parent} && echo 'DIR_FOUND' || echo 'DIR_MISSING'"
    result = _exec(sandbox, check_parent, timeout=10)

    if "DIR_MISSING" in result:
        return f"Error: Parent directory does not exist: {parent}"

    # Create file using heredoc
    # Use base64 encoding to handle special characters safely
    import base64

    content_b64 = base64.b64encode(content.encode()).decode()
    cmd = f"echo '{content_b64}' | base64 -d > {path}"

    output = _exec(sandbox, cmd, timeout=timeout)
    if "Error" in output:
        return output

    return f"Successfully created {path}"


def execute_submit_code(
    sandbox: Sandbox,
    args: dict[str, Any],
    timeout: float = 30.0,
) -> tuple[str, str | None, str | None]:
    """Execute submit tool in CODE mode - agent submits train.py only.

    In CODE mode, the evaluator will:
    1. Upload train.py to container
    2. Execute it to generate submission.csv
    3. Score the generated CSV

    Returns:
        Tuple of (output_message, train_content, None)
        - output_message: Status message for the agent
        - train_content: Content of train.py (for evaluator to execute)
        - None: No submission_path in CODE mode
    """
    # Support both "path" (CODE mode) and "train_path" (backwards compat)
    train_path = args.get("path") or args.get("train_path", "/workspace/train.py")

    # Validate train_path
    if not train_path.endswith(".py"):
        return f"Error: path must be a .py file, got: {train_path}", None, None

    # Check train.py exists
    check_train = f"test -f {train_path} && echo 'EXISTS' || echo 'NOT_EXISTS'"
    result = _exec(sandbox, check_train, timeout=10)
    if "NOT_EXISTS" in result:
        return f"Error: Training script not found: {train_path}", None, None

    # Fetch train.py content
    train_content = _exec(sandbox, f"cat {train_path}", timeout=timeout)
    if "Error" in train_content:
        return f"Error reading train.py: {train_content}", None, None

    return "Solution submitted for evaluation. The evaluator will execute your training script.", train_content, None


def execute_submit_csv(
    sandbox: Sandbox,
    args: dict[str, Any],
    timeout: float = 30.0,
) -> tuple[str, str | None, str | None]:
    """Execute submit tool in CSV mode - agent submits train.py + submission.csv.

    In CSV mode, the agent has already run train.py and generated submission.csv.
    The evaluator will just fetch and score the existing CSV.

    Returns:
        Tuple of (output_message, train_content, submission_path)
        - output_message: Status message for the agent
        - train_content: Content of train.py (for recording)
        - submission_path: Path to submission.csv (for evaluator to fetch)
    """
    train_path = args.get("train_path", "/workspace/train.py")
    submission_path = args.get("submission_path", "/workspace/submission.csv")

    # Validate train_path
    if not train_path.endswith(".py"):
        return f"Error: train_path must be a .py file, got: {train_path}", None, None

    # Validate submission_path
    if not submission_path.endswith(".csv"):
        return f"Error: submission_path must be a .csv file, got: {submission_path}", None, None

    # Check train.py exists
    check_train = f"test -f {train_path} && echo 'EXISTS' || echo 'NOT_EXISTS'"
    result = _exec(sandbox, check_train, timeout=10)
    if "NOT_EXISTS" in result:
        return f"Error: Training script not found: {train_path}", None, None

    # Check submission.csv exists
    check_submission = f"test -f {submission_path} && echo 'EXISTS' || echo 'NOT_EXISTS'"
    result = _exec(sandbox, check_submission, timeout=10)
    if "NOT_EXISTS" in result:
        return f"Error: Submission file not found: {submission_path}. Make sure you have run your training script first.", None, None

    # Fetch train.py content
    train_content = _exec(sandbox, f"cat {train_path}", timeout=timeout)
    if "Error" in train_content:
        return f"Error reading train.py: {train_content}", None, None

    return "Solution submitted for evaluation.", train_content, submission_path


# Backward compatibility alias
def execute_submit(
    sandbox: Sandbox,
    args: dict[str, Any],
    timeout: float = 30.0,
    submit_file: str = "csv",
) -> tuple[str, str | None, str | None]:
    """Execute submit tool - routes to CSV or CODE mode.

    Args:
        sandbox: Sandbox instance
        args: Tool arguments
        timeout: Timeout for file operations
        submit_file: "csv" or "code" mode

    Returns:
        Tuple of (output_message, train_content, submission_path)
    """
    if submit_file == "code":
        return execute_submit_code(sandbox, args, timeout)
    else:
        return execute_submit_csv(sandbox, args, timeout)


def execute_check_submission_validity(
    sandbox: Sandbox,
    args: dict[str, Any],
    task_id: str,
    mle_bench_data_dir: str = "/checkpoint/maui/shared/cache/dojo/tasks/mlebench",
    timeout: float = 60.0,
) -> str:
    """Execute check_submission_validity tool.

    Validates that:
    1. train.py exists
    2. submission.csv has correct format for the competition
    """
    train_path = args.get("train_path", "/workspace/train.py")
    submission_path = args.get("submission_path", "/workspace/submission.csv")

    # Check train.py exists
    check_train = f"test -f {train_path} && echo 'EXISTS' || echo 'NOT_EXISTS'"
    result = _exec(sandbox, check_train, timeout=10)
    if "NOT_EXISTS" in result:
        return f"Validation FAILED: Training script not found: {train_path}"

    # Check submission.csv exists
    check_submission = f"test -f {submission_path} && echo 'EXISTS' || echo 'NOT_EXISTS'"
    result = _exec(sandbox, check_submission, timeout=10)
    if "NOT_EXISTS" in result:
        return f"Validation FAILED: Submission file not found: {submission_path}"

    # Fetch submission.csv to local for validation
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        local_submission_path = f.name

    try:
        # Try to fetch the file
        has_file = sandbox.fetch_file(submission_path, local_submission_path)
        if not has_file:
            return f"Validation FAILED: Could not fetch {submission_path}"

        # Validate using mlebench
        try:
            from mlebench.grade import validate_submission
            from mlebench.registry import registry

            new_registry = registry.set_data_dir(Path(mle_bench_data_dir))
            competition = new_registry.get_competition(task_id)
            is_valid, msg = validate_submission(Path(local_submission_path), competition)

            if is_valid:
                return f"Validation PASSED: {msg}"
            else:
                return f"Validation FAILED: {msg}"

        except ImportError:
            # mlebench not available - do basic validation
            import csv

            try:
                with open(local_submission_path) as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    row_count = sum(1 for _ in reader)
                return f"Validation (basic): CSV has {len(header)} columns, {row_count} rows."
            except Exception as e:
                return f"Validation FAILED: Could not parse CSV: {e}"

    finally:
        import os

        try:
            os.unlink(local_submission_path)
        except Exception:
            pass


def execute_tool(
    sandbox: Sandbox,
    tool_name: str,
    args: dict[str, Any],
    task_id: str = "",
    session_timeout: float = 300.0,
    mle_bench_data_dir: str = "/checkpoint/maui/shared/cache/dojo/tasks/mlebench",
    submit_file: str = "csv",
) -> tuple[str, bool, str | None]:
    """Execute a tool and return the result.

    Args:
        sandbox: Sandbox instance
        tool_name: Name of the tool to execute
        args: Tool arguments
        task_id: Task ID for validation
        session_timeout: Timeout for bash commands
        mle_bench_data_dir: Path to MLE-bench data
        submit_file: "csv" or "code" mode for submit tool

    Returns:
        Tuple of (output, is_terminal, pred_solution):
        - output: Tool output message
        - is_terminal: Whether this ends the rollout (submit)
        - pred_solution: train.py content if submitted, else None
    """
    if tool_name == "bash":
        output = execute_bash(sandbox, args, timeout=session_timeout)
        return output, False, None

    elif tool_name == "edit":
        output = execute_edit(sandbox, args)
        return output, False, None

    elif tool_name == "create":
        output = execute_create(sandbox, args)
        return output, False, None

    elif tool_name == "submit":
        output, train_content, _ = execute_submit(sandbox, args, submit_file=submit_file)
        is_terminal = "Error" not in output
        return output, is_terminal, train_content

    elif tool_name == "check_submission_validity":
        output = execute_check_submission_validity(sandbox, args, task_id, mle_bench_data_dir)
        return output, False, None

    else:
        return f"Unknown tool: {tool_name}. Available: bash, edit, create, submit, check_submission_validity", False, None
