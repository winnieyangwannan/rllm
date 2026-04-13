# Plan: Add MLE-bench Support to rLLM

## Context

The amaia-collab codebase has an MLE-bench environment
(`apps/sea/envs/envs/mle_bench/`) that trains agents to solve Kaggle ML
competitions. The agent interacts with a GPU container (AgentBox) via
bash/edit/submit tools, writes a `solution.py`, and is scored by percentile
on the Kaggle leaderboard using the `mlebench` library.

We want to port this capability to rLLM, following rLLM's existing
`agenthub/swe_agent/` pattern exactly — it's the closest structural analogue
(sandboxed, multi-turn, bash-based, with post-hoc evaluation).

The `agentbox` library is now a **standalone pip-installable package**
(https://github.com/winnieyangwannan/agentbox), which provides a clean 3-tier
API: `AgentBoxManager → Machine → Container → Shell/Notebook/FileOps`.
This eliminates the need for any cross-repo imports from amaia-collab.

## Architecture: Amaia → rLLM Mapping

| Amaia Component | rLLM Equivalent | Notes |
|---|---|---|
| `MLEBenchBashEnv.start()` + `step()` | `MLEBenchAgentFlow.run()` | Single method with internal loop |
| `AgentBoxBackend` | `AgentBoxSandbox` (new `Sandbox` impl) | Wraps standalone `agentbox` package |
| `evaluation.py` (7-stage grading) | `MLEBenchEvaluator.evaluate()` | Post-hoc, accesses sandbox via `episode.artifacts["_sandbox"]` |
| `MLEBenchRewardFn` | Part of evaluator → `EvalOutput(reward=percentile)` | |
| XML/JSON tool parsing | OpenAI native function calling | rLLM agents use `tools=` param on `chat.completions.create` |
| `prompts/` module | `prompts.py` | Port system + instance prompts |
| `samples_per_prompt` | `rollout.n` (group_size) | Same concept |
| `MLEBenchBashConfig` | Constructor args + Hydra config | Flatten into `MLEBenchAgentFlow.__init__` params |

## Dependencies

```toml
[project]
name = "mle-bench-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "rllm",
    "openai",
    "agentbox",           # pip install agentbox — standalone package
    "pandas",
    "numpy",
]

[project.optional-dependencies]
grading = [
    "mlebench",           # For percentile scoring in evaluator
]
training = [
    "hydra-core",
]
```

## Files to Create

### 1. `rllm/sdk/sandbox/backends/agentbox_backend.py` — AgentBox Sandbox

Implements the `Sandbox` protocol (`rllm/sdk/sandbox/protocol.py`) using the
standalone `agentbox` package.

```python
from agentbox import AgentBoxManager, ContainerConfig, HOST2CONTAINER, CONTAINER2HOST

class AgentBoxSandbox:
    """Sandbox protocol implementation backed by the agentbox package.

    Manages the full lifecycle: Manager → Machine → Container.
    The container runs on an H200 GPU node with the competition data
    bind-mounted read-only at /root/data/.
    """

    def __init__(
        self,
        name: str,
        manager_uri: str,
        container_config: ContainerConfig | None = None,
        # Convenience params (used if container_config is None):
        image: str = "",
        superimage_directory: str = "",
        superimage_version: str = "",
        read_only_overlays: list[str] | None = None,
        read_only_binds: dict[str, str] | None = None,
        working_dir: str = "/workspace",
        env: dict[str, str] | None = None,
        max_streaming_blocks: int = 100,
    ):
        self._manager = AgentBoxManager(manager_uri)
        self._machine = self._manager.start_machine(name=name, blocking=True)
        self._max_streaming_blocks = max_streaming_blocks

        if container_config is None:
            container_config = ContainerConfig(
                image_handle=image,
                superimage_directory=superimage_directory,
                superimage_version=superimage_version,
                read_only_overlays=read_only_overlays or [],
                read_only_binds=read_only_binds or {},
                working_dir=working_dir,
                container_runtime="apptainer",
                env=env or {},
            )

        self._container = self._machine.start_container(
            config=container_config, name=name
        )

    # ── Sandbox protocol methods ─────────────────────────────────

    def exec(self, command: str, timeout: float | None = None) -> str:
        """Execute command via streaming shell. Buffers up to
        max_streaming_blocks output blocks, then truncates.
        On 'Killed' in output, queries dmesg for OOM context."""
        from pathlib import Path

        output_parts = []
        block_count = 0

        with self._container.shell(work_dir=Path("/workspace")) as shell:
            for block in shell.execute(command):
                if block_count < self._max_streaming_blocks:
                    output_parts.append(block.output)
                block_count += 1

        result = "".join(output_parts)

        # OOM detection (ported from AMAIA agentbox_backend.py)
        if "Killed" in result:
            try:
                oom_parts = []
                with self._container.shell() as shell:
                    for block in shell.execute("dmesg | tail -5"):
                        oom_parts.append(block.output)
                oom_info = "".join(oom_parts)
                result += f"\n[OOM detected] dmesg:\n{oom_info}"
            except Exception:
                pass

        return result

    def upload_file(self, local_path: str, remote_path: str) -> None:
        self._container.copy_file(local_path, remote_path, HOST2CONTAINER)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload directory by tarring, uploading, and extracting."""
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            tar_path = f.name
        try:
            subprocess.run(
                ["tar", "-czf", tar_path, "-C", local_path, "."],
                check=True,
            )
            remote_tar = f"{remote_path}/__upload.tar.gz"
            self.upload_file(tar_path, remote_tar)
            self.exec(f"cd {remote_path} && tar -xzf __upload.tar.gz && rm __upload.tar.gz")
        finally:
            os.unlink(tar_path)

    def start_agent_process(self, command: str, port: int) -> None:
        raise NotImplementedError("AgentBox sandbox does not support long-running processes")

    def get_endpoint(self, port: int) -> tuple[str, dict[str, str]]:
        raise NotImplementedError("AgentBox sandbox does not support port endpoints")

    def close(self) -> None:
        """Release container and machine resources."""
        try:
            self._machine.free_container(self._container)
        except Exception:
            pass
        try:
            self._manager.free_machine(self._machine)
        except Exception:
            pass

    # ── Extra methods for evaluation ─────────────────────────────

    def fetch_file(self, remote_path: str, local_path: str) -> bool:
        """Download a file from the container to the host."""
        try:
            self._container.copy_file(remote_path, local_path, CONTAINER2HOST)
            return True
        except Exception:
            return False

    def list_directory(self, path: str) -> list:
        """List files in a container directory."""
        return self._container.list_directory(path)
```

#### Register in `create_sandbox()` factory

Add to `rllm/experimental/agents/sandboxed_agent.py` `create_sandbox()`:

```python
elif backend == "agentbox":
    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox
    return AgentBoxSandbox(name=name, **kwargs)
```

### 2. `agenthub/mle_bench_agent/` — Plugin Directory

Structure mirrors `agenthub/swe_agent/`:

```
agenthub/mle_bench_agent/
├── pyproject.toml
└── mle_bench_agent/
    ├── __init__.py
    ├── agent.py       # MLEBenchAgentFlow
    ├── evaluator.py   # MLEBenchEvaluator
    └── prompts.py     # System/instance/error prompts + tool schemas
```

### 3. `agenthub/mle_bench_agent/mle_bench_agent/agent.py` — Agent Flow

Follows `swe_agent/agent.py` pattern exactly:

```python
class MLEBenchAgentFlow(SandboxedAgentFlow):
    max_concurrent: int = 4
    sandbox_backend: str = "agentbox"

    def __init__(
        self,
        manager_uri: str = "",
        max_turns: int = 128,
        session_timeout: float = 360.0,       # Per-bash-call timeout (seconds)
        eval_timeout: int = 300,               # Solution execution timeout
        rollout_timeout: int = 32400,          # Total rollout budget (9 hours)
        data_base_path: str = "",              # Base path for competition data
        superimage_directory: str = "",
        superimage_version: str = "",
        read_only_overlays: list[str] | None = None,
        context_size: int = 131072,
        think: bool = True,
        check_submission_validity: bool = False,
        **kwargs,
    ):
        ...

    def setup_sandbox(self, task: dict, config) -> None:
        """Create AgentBoxSandbox with task-specific data mount.

        Uses ContainerConfig from the agentbox package:
        - superimage_directory / superimage_version for the container image
        - read_only_overlays for cached package layers
        - read_only_binds to mount competition data at /root/data/ (read-only)
        - env: HF_HUB_OFFLINE=1, NLTK_DATA=/root/.nltk_data
        - working_dir: /workspace
        """
        from agentbox import ContainerConfig

        task_id = task.get("task_id", task.get("instance_id", "unknown"))
        data_path = f"{self.data_base_path}/{task_id}/prepared/public"

        container_config = ContainerConfig(
            superimage_directory=self.superimage_directory,
            superimage_version=self.superimage_version,
            container_runtime="apptainer",
            read_only_overlays=self.read_only_overlays or [],
            read_only_binds={data_path: "/root/data"},
            working_dir="/workspace",
            env={"HF_HUB_OFFLINE": "1", "NLTK_DATA": "/root/.nltk_data"},
        )

        name = f"rllm-mle-{task_id}-{uuid.uuid4().hex[:6]}"

        from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox
        self._sandbox = AgentBoxSandbox(
            name=name,
            manager_uri=self.manager_uri,
            container_config=container_config,
        )

        self.on_sandbox_ready(task, config)

    def on_sandbox_ready(self, task: dict, config) -> None:
        """Ensure /workspace exists."""
        _safe_exec(self.sandbox, "mkdir -p /workspace")

    def run(self, task: dict, config) -> Episode:
        """Run the multi-turn agent loop.

        1. Build messages: [system_prompt, instance_prompt(task_description + data_info)]
        2. Create OpenAI client via config.base_url (Model Gateway)
        3. Run _run_agent_loop() with native function calling
        4. Return Episode with Trajectory(name="mle_solver")
        """
        task_id = task.get("task_id", task.get("instance_id", "unknown"))

        # Gather data info from the container
        data_info = _safe_exec(self.sandbox, DATA_INFO_COMMAND)

        # Build initial messages
        system_prompt = SYSTEM_PROMPT.format(
            timeout_min=int(self.session_timeout / 60),
            context_size=self.context_size,
            max_turns=self.max_turns,
            eval_timeout_hrs=int(self.eval_timeout / 3600),
        )
        instance_prompt = INSTANCE_PROMPT.format(
            task_description=task.get("task_description", ""),
            data_info=data_info,
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
mle_bench_agent = MLEBenchAgentFlow()
```

#### `_run_agent_loop()` — standalone function

Same structure as `swe_agent/agent.py` `_run_agent_loop()` (lines 71–167):

```python
def _run_agent_loop(
    client: openai.OpenAI,
    model: str,
    messages: list[dict],
    sandbox: Sandbox,
    max_turns: int = 128,
    session_timeout: float = 360.0,
    rollout_timeout: int = 32400,
    temperature: float = 1.0,
    check_submission_validity: bool = False,
) -> tuple[list[Step], list[dict], str | None]:
    """Custom agent loop with format error recovery and submission detection.

    Key differences from swe_agent loop:
    - Tools: BASH_TOOL, SUBMIT_TOOL, and optionally CHECK_SUBMISSION_TOOL
    - Submit detection: fn_name == "submit" reads solution.py and breaks
    - Rollout timeout: checks wall-clock time budget each turn
    - Temperature: 1.0 (for RL exploration), not 0.0
    """
    tool_schemas = [BASH_TOOL, SUBMIT_TOOL]
    if check_submission_validity:
        tool_schemas.append(CHECK_SUBMISSION_TOOL)

    steps: list[Step] = []
    format_errors = 0
    max_format_errors = 3
    pred_solution = None
    start_time = time.time()

    for turn in range(max_turns):
        # Check rollout timeout
        if time.time() - start_time >= rollout_timeout:
            break

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
            temperature=temperature,
        )
        choice = response.choices[0]
        msg = choice.message
        messages.append(msg.model_dump(exclude_none=True))

        # No tool calls — format error recovery (same as swe_agent)
        if not msg.tool_calls:
            format_errors += 1
            if format_errors >= max_format_errors:
                break
            messages.append({"role": "user", "content": FORMAT_ERROR_MSG})
            continue

        format_errors = 0
        submitted = False

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"command": tc.function.arguments}

            if fn_name == "bash":
                command = args.get("command", "")
                output = _truncate_output(
                    _safe_exec(sandbox, command, timeout=session_timeout)
                )

            elif fn_name == "submit":
                # Read the solution file the agent is submitting
                solution_path = args.get("path", "/workspace/solution.py")
                pred_solution = _safe_exec(
                    sandbox, f"cat {solution_path}", timeout=30
                )
                output = "Solution submitted for evaluation."
                submitted = True

            elif fn_name == "check_submission_validity":
                # Run solution.py and check submission.csv format
                output = _truncate_output(
                    _safe_exec(sandbox, CHECK_SUBMISSION_COMMAND, timeout=session_timeout)
                )

            else:
                output = f"Unknown tool: {fn_name}. Available: bash, submit."

            steps.append(Step(input=args, output=output))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

        if submitted:
            return steps, messages, pred_solution

    return steps, messages, pred_solution
```

### 4. `agenthub/mle_bench_agent/mle_bench_agent/evaluator.py` — Evaluation

Follows `swe_agent/evaluator.py` pattern:

```python
class MLEBenchEvaluator:
    def __init__(self, eval_timeout: int = 300):
        self.eval_timeout = eval_timeout

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        """7-stage evaluation pipeline.

        Accesses the SAME sandbox used during rollout via
        episode.artifacts["_sandbox"] (set by EvalRunner).
        This avoids spinning up a new container.
        """
        sandbox = episode.artifacts.get("_sandbox")
        task_id = task.get("task_id", task.get("instance_id"))
        pred_solution = episode.artifacts.get("pred_solution")

        # Stage 1: Sanity check — solution exists, references submission.csv
        if not pred_solution:
            return _fail("No solution submitted", task_id)
        if "submission.csv" not in pred_solution:
            return _fail("Solution does not reference submission.csv", task_id)

        # Stage 2: Setup — remove any stale submission.csv
        _safe_exec(sandbox, "rm -f /workspace/submission.csv")

        # Stage 3: Solution already written to /workspace/solution.py by agent

        # Stage 4: Execute solution
        exec_output = _safe_exec(
            sandbox,
            "cd /workspace && python solution.py",
            timeout=self.eval_timeout,
        )

        # Stage 5: Fetch submission.csv to local temp dir
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            local_csv = f"{tmpdir}/submission.csv"
            if not sandbox.fetch_file("/workspace/submission.csv", local_csv):
                return _fail("submission.csv not found after execution", task_id,
                             signals=[Signal("exec_output", exec_output)])

            # Stage 6: Validate CSV format (lazy import mlebench)
            try:
                from mlebench.grade import validate_submission
                valid, validation_msg = validate_submission(task_id, local_csv)
                if not valid:
                    return _fail(f"Invalid CSV: {validation_msg}", task_id)
            except ImportError:
                pass  # Skip validation if mlebench not installed

            # Stage 7: Score — compute percentile rank
            try:
                from mlebench.evaluate import evaluate_submission
                raw_score = evaluate_submission(task_id, local_csv)
                percentile = get_rank_and_percentile(task_id, raw_score)
            except Exception as e:
                return _fail(f"Scoring error: {e}", task_id)

        return EvalOutput(
            reward=percentile,
            is_correct=percentile > 0.0,
            signals=[
                Signal("percentile", percentile),
                Signal("raw_score", raw_score),
                Signal("valid_submission", 1.0),
                Signal("submission_provided", 1.0),
            ],
        )
```

Key details:
- Accesses sandbox via `episode.artifacts["_sandbox"]` (set by EvalRunner)
- Uses `mlebench` library via lazy import with `try/except`
- On any failure → `EvalOutput(reward=0.0)` with descriptive signals
- `get_rank_and_percentile()` ported from AMAIA's `evaluation.py`:
  uses tolerance-based comparison (`rel_tol=1e-9`, `abs_tol=1e-12`)
  against the competition leaderboard

### 5. `agenthub/mle_bench_agent/mle_bench_agent/prompts.py` — Prompts

Port from `amaia-collab/apps/sea/envs/envs/mle_bench/prompts/common.py` and `gpt5.py`:

```python
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


INSTANCE_PROMPT = """<task_description>
{task_description}
</task_description>

<data_info>
{data_info}
</data_info>

Solve the task following the guidelines above. When you are confident your
solution is correct, use the submit tool with the path to your solution.py.
""".strip()


FORMAT_ERROR_MSG = (
    "You must use one of the available tools (bash or submit) to interact "
    "with the environment. Please try again with a valid tool call."
)


# OpenAI function-calling tool schemas

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
        "description": (
            "Submit your final solution. The solution.py file must write "
            "predictions to /workspace/submission.csv when executed."
        ),
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
        "description": (
            "Run solution.py and validate that the generated submission.csv "
            "matches the expected format (column names, row count, value ranges)."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


# Helper constants

DATA_INFO_COMMAND = '''cd /root/data && \
echo "=== DATA STRUCTURE ===" && ls -sh && \
echo -e "\\n=== CSV ROW COUNTS ===" && wc -l *.csv 2>/dev/null && \
echo -e "\\n=== SAMPLE SUBMISSION FORMAT ===" && head -3 sample_submission.csv 2>/dev/null'''

CHECK_SUBMISSION_COMMAND = (
    "cd /workspace && python solution.py && "
    "echo '=== SUBMISSION HEAD ===' && head -5 submission.csv && "
    "echo '=== SUBMISSION SHAPE ===' && wc -l submission.csv"
)
```

### 6. `agenthub/mle_bench_agent/pyproject.toml`

```toml
[project]
name = "mle-bench-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "rllm",
    "openai",
    "agentbox",
    "pandas",
    "numpy",
]

[project.optional-dependencies]
grading = ["mlebench"]
training = ["hydra-core"]
```

## Configuration

Default training config (Hydra YAML or passed programmatically):

```yaml
backend: verl

model:
  name: Qwen/Qwen3-8B
  lora_rank: 32

training:
  group_size: 4           # Rollouts per task for GRPO
  batch_size: 8
  epochs: 5
  learning_rate: 1e-5

algorithm:
  estimator: grpo
  kl_coef: 0.01

mle_bench:
  agentbox_manager_uri: "http://agentbox-manager:8080"
  max_turns: 128
  session_timeout: 360
  eval_timeout: 300
  rollout_timeout: 32400   # 9 hours
  context_size: 131072
  think: true
  check_submission_validity: false
  # Container config
  superimage_directory: "/checkpoint/maui_sft/shared/sif"
  superimage_version: "2025-05-02v2"
  read_only_overlays:
    - "/checkpoint/fair-maui-hs/hotfix/kniu.2025-09-19.cache.overlay.ext3.img"
  data_base_path: "/checkpoint/maui/shared/cache/dojo/tasks/mlebench"
```

## Implementation & Testing Plan

Each step below is independently testable. Complete and verify each step before
moving on. The order follows the dependency chain — later steps build on earlier
ones.

### Step 1: AgentBox Raw Connection Test _(no rLLM code needed)_

**Goal**: Verify the standalone `agentbox` pip package works end-to-end.

**What to build**: A standalone test script (no rLLM imports).

**What to test**:
- A) Connect to the AgentBox manager and start/stop a machine
- B) Start a container with a basic `ContainerConfig` and run `echo hello`
- C) Verify streaming shell output works (multi-block output)
- D) Test `copy_file` in both directions (host→container, container→host)
- E) Verify `free_container` and `free_machine` release resources cleanly

**Example test script**:
```python
from agentbox import AgentBoxManager, ContainerConfig

mgr = AgentBoxManager("http://your-manager:8080")
machine = mgr.start_machine(name="test-conn", blocking=True)
config = ContainerConfig(
    superimage_directory="...",
    superimage_version="...",
    container_runtime="apptainer",
    working_dir="/workspace",
)
container = machine.start_container(config=config, name="test-conn")

# Test exec
with container.shell() as shell:
    for block in shell.execute("echo hello && whoami"):
        print(block.output)

# Test file round-trip
# ... upload a file, download it, compare

machine.free_container(container)
mgr.free_machine(machine)
print("All connection tests passed.")
```

**Pass criteria**: Script runs without errors, prints "hello", and cleans up.

---

### Step 2: AgentBoxSandbox Wrapper _(first rLLM code)_

**Goal**: Wrap the raw agentbox API in the rLLM `Sandbox` protocol.

**What to build**:
1. Create `rllm/sdk/sandbox/backends/agentbox_backend.py` — the `AgentBoxSandbox` class
2. Register `"agentbox"` backend in `create_sandbox()` factory

**What to test**:
- A) `AgentBoxSandbox.exec("echo hello")` returns `"hello\n"`
- B) `exec()` with a long-running command respects streaming block limit
- C) OOM detection: run a command that outputs "Killed", verify dmesg is appended
- D) `upload_file()` / `fetch_file()` round-trip: upload a file, fetch it back, compare contents
- E) `upload_dir()` works for a small directory
- F) `close()` releases resources without error
- G) `create_sandbox("agentbox", name="test", manager_uri="...")` returns a working instance

**Pass criteria**: All Sandbox protocol methods behave correctly; factory registration works.

---

### Step 3: Prompts & Tool Schemas _(pure data, no infrastructure)_

**Goal**: Validate prompts and OpenAI tool schemas are correct before using them.

**What to build**:
1. Create `agenthub/mle_bench_agent/mle_bench_agent/prompts.py` with all constants

**What to test**:
- A) `SYSTEM_PROMPT.format(...)` renders correctly with sample values
- B) `INSTANCE_PROMPT.format(...)` renders correctly with sample task data
- C) Pass `BASH_TOOL`, `SUBMIT_TOOL`, `CHECK_SUBMISSION_TOOL` to a real
     `client.chat.completions.create(tools=...)` call — confirm no schema validation errors
- D) Verify `DATA_INFO_COMMAND` and `CHECK_SUBMISSION_COMMAND` are valid shell commands
     (run them in a container from Step 2)

**Pass criteria**: All templates render, all tool schemas are accepted by the OpenAI API.

---

### Step 4: Agent Loop with Mock Sandbox _(logic testing, no GPU needed)_

**Goal**: Test `_run_agent_loop()` logic without a real container or GPU.

**What to build**:
1. Create `agenthub/mle_bench_agent/mle_bench_agent/agent.py` with `_run_agent_loop()`
2. Create a `MockSandbox` class that returns canned responses

**What to test**:
- A) **Normal flow**: Mock sandbox returns bash output → LLM calls bash → LLM calls submit → loop exits with `pred_solution`
- B) **Format error recovery**: Simulate LLM returning no tool calls → verify `FORMAT_ERROR_MSG` is appended, loop retries (up to 3 times)
- C) **Rollout timeout**: Set `rollout_timeout=1` second, verify loop exits on time
- D) **Output truncation**: Mock a very long bash output, verify `_truncate_output` truncates it
- E) **Unknown tool**: Simulate LLM calling a nonexistent tool → verify error message returned
- F) **JSON decode error**: Simulate malformed `function.arguments` → verify graceful fallback

**Pass criteria**: All logic paths tested; no real infrastructure required.

---

### Step 5: Agent Loop with Real Sandbox _(integration test)_

**Goal**: Run the full agent loop against a real AgentBox container and LLM.

**What to build**:
1. Complete `MLEBenchAgentFlow` with `setup_sandbox()`, `on_sandbox_ready()`, and `run()`
2. Wire up to a real LLM via Model Gateway

**What to test**:
- A) Pick **one simple competition** (small tabular dataset, fast training)
- B) Run a single rollout with `max_turns=10` — verify the loop completes:
     LLM call → bash tool → LLM → ... → submit
- C) Verify the `Episode` output has correct structure: `trajectories`, `artifacts["pred_solution"]`
- D) Confirm sandbox cleanup (`close()`) runs after the rollout

**Pass criteria**: Agent produces a `solution.py` and submits it (quality doesn't matter yet).

---

### Step 6: Evaluator with Known Submissions _(test grading in isolation)_

**Goal**: Test the 7-stage evaluation pipeline against pre-made submissions.

**What to build**:
1. Create `agenthub/mle_bench_agent/mle_bench_agent/evaluator.py` — `MLEBenchEvaluator`
2. Port `get_rank_and_percentile()` from AMAIA

**What to test**:
- A) **Stage 1 — no solution**: Pass an episode with `pred_solution=None` → `reward=0.0`
- B) **Stage 1 — no submission.csv ref**: Pass a solution without "submission.csv" → `reward=0.0`
- C) **Stage 4 — execution failure**: Pre-place a broken `solution.py` in the container → verify exec error captured in signals
- D) **Stage 5 — missing CSV**: Solution runs but doesn't produce `submission.csv` → `reward=0.0`
- E) **Stage 6 — invalid CSV format**: Place a malformed `submission.csv` → verify validation catches it
- F) **Stage 7 — known good submission**: Place a pre-computed correct `submission.csv`
     for a known competition → verify percentile matches expected value
- G) Verify `get_rank_and_percentile()` tolerance logic with edge-case scores

**Pass criteria**: Each failure mode returns `reward=0.0` with correct signals; known good submissions score correctly.

---

### Step 7: End-to-End Eval Run _(agent + evaluator together)_

**Goal**: Full pipeline — agent produces a submission, evaluator grades it.

**What to build**:
1. Create `agenthub/mle_bench_agent/pyproject.toml`
2. Create plugin `__init__.py` with entry points
3. Wire agent + evaluator together

**What to test**:
- A) Run agent on one competition → get Episode → pass to evaluator → get `EvalOutput`
- B) Verify the sandbox is reused (evaluator accesses via `episode.artifacts["_sandbox"]`)
- C) Run on 2–3 competitions to confirm generalization
- D) Verify resource cleanup: no leaked machines/containers after the run

**Pass criteria**: End-to-end produces a valid `EvalOutput` with a non-trivial percentile score.

---

### Step 8: Training Integration _(final step)_

**Goal**: Train a model using RL on MLE-bench tasks.

**What to build**:
1. Create training script using `UnifiedTrainer`
2. Create Hydra config YAML

**What to test**:
- A) Test with **tinker backend** (single machine) first — 1 epoch, 2 tasks, `group_size=2`
- B) Verify GRPO advantage calculation works with percentile rewards
- C) Test with **verl backend** (distributed) — same small config
- D) Verify checkpoint saving/loading
- E) Run a multi-epoch training loop and confirm loss decreases

**Pass criteria**: Training loop completes without error; model checkpoint is saved.

---

### Priority & Dependency Matrix

| Step | Component | Depends On | Infra Needed | Complexity |
|------|-----------|-----------|-------------|------------|
| 1 | AgentBox raw connection | `agentbox` package | AgentBox manager | Low |
| 2 | `AgentBoxSandbox` wrapper | Step 1 | AgentBox manager | Low |
| 3 | Prompts & tool schemas | None | OpenAI API key | Trivial |
| 4 | Agent loop (mocked) | Step 3 | None | Medium |
| 5 | Agent loop (real) | Steps 2, 3, 4 | AgentBox + LLM | High |
| 6 | Evaluator (staged) | Step 2 | AgentBox + `mlebench` | Medium |
| 7 | End-to-end eval | Steps 5, 6 | AgentBox + LLM + `mlebench` | High |
| 8 | Training integration | Step 7 | All above + GPU cluster | High |

**Key principle**: Test each layer in isolation before combining. Steps 1–4 can
be done with minimal infrastructure. Steps 3 and 4 can be done in parallel with
Steps 1–2 since they have no dependency on AgentBox infrastructure.

## Open Questions

1. **`mlebench` grading library** — Is this pip-installable, or does it still
   live inside AMAIA? If it's not a standalone package, the evaluator will need
   a vendored copy of the grading functions (`validate_submission`,
   `evaluate_submission`, `get_rank_and_percentile`).

2. **Task dataset** — How are competition task definitions (task_id,
   task_description, data paths) loaded? Need to confirm whether rLLM's
   `DatasetRegistry` supports this or if we load from a JSONL file directly.

3. **Container reuse across group_size rollouts** — For GRPO with `group_size=4`,
   should all 4 rollouts share a container (same machine, different runs) or
   each get their own? Sharing saves startup time but risks state leakage.
