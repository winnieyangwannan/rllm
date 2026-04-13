"""MLE-bench agent flow.

This module implements the MLEBenchAgentFlow class which runs multi-turn
agent interactions to solve Kaggle ML competitions in sandboxed GPU containers.

Structure follows agenthub/swe_agent/:
- Custom agent loop with format error recovery
- Submit tool detection to end rollout
- Smart output truncation to avoid context overflow
- Rollout timeout for total wall-clock budget

Tools (matching amaia-collab mle_bench_bash_env_with_csv_check):
- bash: Execute commands
- edit: Search/replace in files
- create: Create new files
- submit: Submit train.py + submission.csv
- check_submission_validity: Validate CSV format
"""

from __future__ import annotations

import json
import logging
import time
import uuid

import openai

from rllm.experimental.agents.sandboxed_agent import SandboxedAgentFlow
from rllm.sdk.sandbox.protocol import Sandbox
from rllm.types import Episode, Step, Trajectory

from .prompts import (
    DATA_INFO_COMMAND,
    FORMAT_ERROR_MSG,
    INSTANCE_PROMPT,
    SYSTEM_PROMPT,
)
from .tools import (
    _exec,
    execute_tool,
    get_tools,
)

logger = logging.getLogger(__name__)


def _run_agent_loop(
    client: openai.OpenAI,
    model: str,
    messages: list[dict],
    sandbox: Sandbox,
    max_turns: int = 128,
    session_timeout: float = 360.0,
    rollout_timeout: float = 32400.0,  # 9 hours default
    temperature: float = 1.0,
    check_submission_validity: bool = True,
    task_id: str = "",
    mle_bench_data_dir: str = "/checkpoint/maui/shared/cache/dojo/tasks/mlebench",
    submit_file: str = "csv",
) -> tuple[list[Step], list[dict], str | None]:
    """Custom agent loop with format error recovery and submission detection.

    Args:
        client: OpenAI client instance
        model: Model name to use
        messages: Initial conversation messages
        sandbox: Sandbox instance for executing commands
        max_turns: Maximum number of LLM turns
        session_timeout: Timeout per bash command (seconds)
        rollout_timeout: Total wall-clock budget (seconds)
        temperature: Sampling temperature (1.0 for RL exploration)
        check_submission_validity: Whether to include check_submission_validity tool
        task_id: Task ID for validation
        mle_bench_data_dir: Path to MLE-bench data directory
        submit_file: "csv" (agent generates CSV) or "code" (evaluator runs train.py)

    Returns:
        Tuple of (steps, messages, pred_solution):
        - steps: List of Step objects for the trajectory
        - messages: Full conversation history
        - pred_solution: Content of train.py if submitted, else None

    Submit Modes:
    - "csv" (default): Agent runs train.py, generates submission.csv, submits both
      Tools: bash, edit, create, submit(train_path, submission_path), check_submission_validity
    - "code": Agent submits just train.py, evaluator executes it
      Tools: bash, edit, create, submit(path)

    Key features:
    - Format error recovery: if LLM responds without tool calls, retry up to 3 times
    - Submit detection: 'submit' tool ends loop and captures train.py
    - Rollout timeout: checks wall-clock time budget each turn
    - Temperature 1.0 for RL exploration (not 0.0 like SWE-agent)
    """
    # Select tool set based on mode
    tool_schemas = get_tools(submit_file=submit_file, check_submission_validity=check_submission_validity)

    steps: list[Step] = []
    format_errors = 0
    max_format_errors = 3
    pred_solution = None
    start_time = time.time()

    for turn in range(max_turns):
        # Check rollout timeout
        elapsed = time.time() - start_time
        if elapsed >= rollout_timeout:
            logger.warning("Rollout timeout reached after %.1f seconds", elapsed)
            break

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schemas,
                temperature=temperature,
            )
        except Exception:
            logger.exception("LLM API call failed on turn %d", turn)
            break

        choice = response.choices[0]
        msg = choice.message

        # Append assistant message to conversation
        messages.append(msg.model_dump(exclude_none=True))

        # No tool calls — send format error and retry
        if not msg.tool_calls:
            content = msg.content or ""
            format_errors += 1

            if format_errors >= max_format_errors:
                logger.warning("Max format errors reached, ending loop")
                steps.append(Step(input=f"turn_{turn}", output=content, done=True))
                return steps, messages, pred_solution

            # Send format error recovery message
            messages.append({"role": "user", "content": FORMAT_ERROR_MSG})
            continue

        # Reset format error counter on successful tool use
        format_errors = 0
        submitted = False

        # Process each tool call
        for tc in msg.tool_calls:
            fn_name = tc.function.name

            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # Fallback: treat the whole string as the command
                args = {"command": tc.function.arguments}

            # Execute tool using tools.py
            output, is_terminal, solution_content = execute_tool(
                sandbox=sandbox,
                tool_name=fn_name,
                args=args,
                task_id=task_id,
                session_timeout=session_timeout,
                mle_bench_data_dir=mle_bench_data_dir,
                submit_file=submit_file,
            )

            if is_terminal:
                submitted = True
                pred_solution = solution_content

            steps.append(Step(input={"tool": fn_name, **args}, output=output))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                }
            )

        if submitted:
            return steps, messages, pred_solution

    return steps, messages, pred_solution


class MLEAgentFlow(SandboxedAgentFlow):
    """AgentFlow for MLE-bench: runs a bash/submit agent in a sandboxed GPU container.

    Sandbox lifecycle is managed by EvalRunner via SandboxedAgentFlow:
    1. setup_sandbox() creates AgentBox sandbox with competition data mounted
    2. on_sandbox_ready() ensures /workspace exists
    3. run() executes the custom agent loop
    4. teardown_sandbox() destroys the container

    Uses AgentBox backend for GPU containers on FAIR cluster.
    """

    max_concurrent: int = 4
    sandbox_backend: str = "agentbox"

    def __init__(
        self,
        manager_uri: str = "",
        max_turns: int = 128,
        session_timeout: float = 360.0,
        eval_timeout: int = 300,
        rollout_timeout: float = 32400.0,
        data_base_path: str = "",
        superimage_directory: str = "",
        superimage_version: str = "",
        read_only_overlays: list[str] | None = None,
        context_size: int = 131072,
        think: bool = True,
        check_submission_validity: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.manager_uri = manager_uri
        self.max_turns = max_turns
        self.session_timeout = session_timeout
        self.eval_timeout = eval_timeout
        self.rollout_timeout = rollout_timeout
        self.data_base_path = data_base_path
        self.superimage_directory = superimage_directory
        self.superimage_version = superimage_version
        self.read_only_overlays = read_only_overlays
        self.context_size = context_size
        self.think = think
        self.check_submission_validity = check_submission_validity

    def setup_sandbox(self, task: dict, config) -> None:
        """Create AgentBoxSandbox with task-specific data mount."""
        from agentbox import ContainerConfig

        from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

        task_id = task.get("task_id", task.get("instance_id", "unknown"))
        data_path = f"{self.data_base_path}/{task_id}/prepared/public"

        container_config = ContainerConfig(
            superimage_directory=self.superimage_directory,
            superimage_version=self.superimage_version,
            container_runtime="apptainer",
            read_only_overlays=self.read_only_overlays or [],
            read_only_binds={data_path: "/root/data"} if self.data_base_path else {},
            working_dir="/workspace",
            env={"HF_HUB_OFFLINE": "1", "NLTK_DATA": "/root/.nltk_data"},
        )

        name = f"rllm-mle-{task_id}-{uuid.uuid4().hex[:6]}"

        self._sandbox = AgentBoxSandbox(
            name=name,
            manager_uri=self.manager_uri,
            container_config=container_config,
        )

        self.on_sandbox_ready(task, config)

    def on_sandbox_ready(self, task: dict, config) -> None:
        """Ensure /workspace exists."""
        if self.sandbox is not None:
            _exec(self.sandbox, "mkdir -p /workspace")

    def run(self, task: dict, config) -> Episode:
        """Run the multi-turn agent loop.

        1. Build messages: [system_prompt, instance_prompt(task_description + data_info)]
        2. Create OpenAI client via config.base_url (Model Gateway)
        3. Run _run_agent_loop() with native function calling
        4. Return Episode with Trajectory(name="mle_solver")
        """
        task_id = task.get("task_id", task.get("instance_id", "unknown"))

        # Gather data info from the container (currently unused, but kept for debugging)
        _data_info = ""
        if self.sandbox is not None:
            _data_info = _exec(self.sandbox, DATA_INFO_COMMAND)

        # Build initial messages
        system_prompt = SYSTEM_PROMPT.format(
            timeout_min=int(self.session_timeout / 60),
            context_size=self.context_size,
            eval_timeout_hrs=int(self.eval_timeout / 3600) or 1,
        )
        instance_prompt = INSTANCE_PROMPT.format(
            task_description=task.get("task_description", ""),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]

        client = openai.OpenAI(base_url=config.base_url, api_key="not-needed")

        steps, messages, pred_solution = _run_agent_loop(
            client=client,
            model=config.model,
            messages=messages,
            sandbox=self.sandbox,
            max_turns=self.max_turns,
            session_timeout=self.session_timeout,
            rollout_timeout=self.rollout_timeout,
            check_submission_validity=self.check_submission_validity,
        )

        trajectory = Trajectory(
            name="mle_solver",
            task=task,
            steps=steps,
            output=pred_solution or "",
        )
        return Episode(
            id=f"{task_id}:0",
            task=task,
            trajectories=[trajectory],
            artifacts={
                "answer": "SUBMITTED" if pred_solution else None,
                "pred_solution": pred_solution,
                "task_id": task_id,
            },
        )


# Module-level singleton for plugin entry point
mle_agent = MLEAgentFlow()
