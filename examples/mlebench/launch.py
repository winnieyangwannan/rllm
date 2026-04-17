#!/usr/bin/env python3
"""SLURM Launcher for MLE-bench Evaluation with Reproducibility Features.

This launcher creates a dump directory, copies code for reproducibility,
and submits a SLURM job to run the evaluation.

Usage:
    cd /home/winnieyangwn/rllm/examples/mlebench

    # Basic usage (64 samples with gpt5 config)
    python launch.py --config configs/gpt5.yaml --name exp_001 --task mlsp-2013-birds

    # Multiple tasks with multi-node
    python launch.py --config configs/gpt5.yaml --name multi_node_2_tasks --tasks mlsp-2013-birds,spooky-author-identification --samples 64 --nodes 2
    python launch.py --config configs/test_code.yaml --name multi_node_2_tasks_code --tasks mlsp-2013-birds,spooky-author-identification --samples 64 --nodes 2
    python launch.py --config configs/replicate_code.yaml --name  replicate_code_v3 --tasks mlsp-2013-birds --samples 64 --nodes 1
    python launch.py --config configs/qwen27b_vllm_code.yaml --name  qwen27b_vllm_test --tasks mlsp-2013-birds --samples 64 --nodes 1


    # Resume a failed/incomplete run
    # - Shows progress for each task (completed / remaining)
    # - Copies fresh code to code/{SLURM_JOB_ID}/ subfolder
    # - Skips already completed samples automatically
    # resume
    python launch.py --config configs/gpt5.yaml --name multi_node_2_tasks --tasks mlsp-2013-birds,spooky-author-identification --samples 64 --nodes 2 --resume

    # start from fresh
    python launch.py --config configs/gpt5.yaml --name multi_node_2_tasks --tasks mlsp-2013-birds,spooky-author-identification --samples 64 --nodes 2


    # Custom samples and time (run from eval_integration directory)
    python launch.py --config configs/gpt5.yaml --name exp_003 --task mlsp-2013-birds --samples 32 --time 12:00:00

    # Dry run (print sbatch script without submitting)
    python launch.py --config configs/gpt5.yaml --name exp_004 --task mlsp-2013-birds --dry-run

Features:
    - Creates dump directory for each experiment
    - Copies code for reproducibility (into code/ or code/{job_id}/ for resume)
    - Copies config file to dump directory
    - Auto-generates sbatch script with proper resource requests
    - Resume mode: checks progress, copies fresh code with job ID, skips completed samples
    - Logs captured in dump directory
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Default paths
RLLM_ROOT = Path("/home/winnieyangwn/rllm")
EVAL_SCRIPT = RLLM_ROOT / "examples/mlebench/eval.py"
DEFAULT_DUMP_DIR = Path("/checkpoint/maui_sft/winnieyangwn/rllm/eval")


def get_completed_samples(jsonl_path: Path) -> set[int]:
    """Read existing JSONL file and return set of completed sample indices.

    Parses sample_idx from episode id format: "{task_id}:{sample_idx}"
    """
    completed = set()
    if not jsonl_path.exists():
        return completed

    try:
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        episode_id = data.get("id", "")
                        if ":" in episode_id:
                            try:
                                sample_idx = int(episode_id.split(":")[-1])
                                completed.add(sample_idx)
                            except ValueError:
                                continue
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Warning: Failed to read {jsonl_path}: {e}")

    return completed


def check_resume_progress(dump_dir: Path, task_ids: list[str], num_samples: int) -> dict[str, dict]:
    """Check progress for each task in resume mode.

    Returns:
        Dict mapping task_id -> {"completed": int, "remaining": int, "total": int}
    """
    results_dir = dump_dir / "results"
    progress = {}

    for task_id in task_ids:
        jsonl_path = results_dir / f"{task_id}.jsonl"
        completed = get_completed_samples(jsonl_path)
        num_completed = len(completed)
        progress[task_id] = {
            "completed": num_completed,
            "remaining": num_samples - num_completed,
            "total": num_samples,
        }

    return progress


def create_dump_directory(name: str, base_dir: Path, dry_run: bool = False) -> Path:
    """Create dump directory."""
    dump_dir = base_dir / name
    if not dry_run:
        dump_dir.mkdir(parents=True, exist_ok=True)
    return dump_dir


def copy_code(dump_dir: Path, source_dir: Path = RLLM_ROOT) -> Path:
    """Copy code to dump directory for reproducibility."""
    code_dir = dump_dir / "code"

    # Patterns to ignore when copying
    ignore_patterns = shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", "*.pyo", ".venv", "venv", "*.egg-info", ".eggs", "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "*.log"
    )

    print(f"Copying code from {source_dir} to {code_dir}...")
    shutil.copytree(source_dir, code_dir, ignore=ignore_patterns)
    print(f"  ✓ Code copied ({sum(1 for _ in code_dir.rglob('*.py'))} Python files)")

    return code_dir


def generate_sbatch_script(
    name: str,
    config_path: Path,
    dump_dir: Path,
    code_dir: Path,
    task: str | None = None,
    tasks: str | None = None,
    samples: int = 64,
    partition: str = "h200",
    qos: str = "h200_coding_shared",
    account: str = "aira_ws2",
    time: str = "168:00:00",
    nodes: int = 1,
    ray: bool = True,
    cpus_per_node: int = 192,
    gpus_per_node: int = 8,
    resume: bool = False,
    source_code_dir: Path | None = None,
) -> str:
    """Generate sbatch script content.

    Args:
        ray: If True (default), use Ray for task distribution via eval_ray.py.
        cpus_per_node: CPUs to register per Ray node (default: 192 for H200 nodes)
        nodes: Number of SLURM nodes. If >1, starts a Ray cluster across nodes.
        resume: If True, copy code at runtime into code/{SLURM_JOB_ID}/ subfolder.
        source_code_dir: Source directory to copy code from (only used when resume=True).

    Multi-node Ray approach:
        When nodes > 1:
        1. Start Ray head on first node (blocking mode)
        2. Start Ray workers on remaining nodes (blocking mode via srun)
        3. Run eval_ray.py on head node which connects to cluster
        4. Ray distributes tasks across all nodes automatically
    """

    # Determine task arguments
    if task:
        task_args = f"--task {task}"
    elif tasks:
        task_args = f"--tasks {tasks}"
    else:
        raise ValueError("Must specify either --task or --tasks")

    # Always use eval_ray.py (works in both single-node and multi-node)
    eval_script = "eval_ray.py" if ray else "eval.py"

    # For resume mode, code_dir will be set at runtime via $CODE_DIR shell variable
    # For normal mode, use the provided code_dir path
    if resume and source_code_dir:
        # Use shell variable that will be set at runtime
        runtime_code_dir = "$CODE_DIR"
    else:
        runtime_code_dir = str(code_dir)

    script = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --mem=0
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --account={account}
#SBATCH --output={dump_dir}/logs/%j.out
#SBATCH --error={dump_dir}/logs/%j.err

echo "========================================"
echo "MLE-bench Evaluation: {name}"
echo "========================================"
echo "Job ID: ${{SLURM_JOB_ID}}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Dump directory: {dump_dir}"
echo "Ray mode: {ray}"
echo "Nodes: {nodes}"
echo "CPUs per node: {cpus_per_node}"
echo "Samples: {samples}"
echo "Resume mode: {resume}"
echo "========================================"

# Activate conda environment and ensure no venv interference
source /storage/home/winnieyangwn/miniforge3/etc/profile.d/conda.sh
conda activate rllm

# Clear any venv settings that might interfere
unset VIRTUAL_ENV
export PATH=$(echo "$PATH" | tr ':' '\\n' | grep -v '.venv' | tr '\\n' ':' | sed 's/:$//')
unset PYTHONPATH

echo "Python: $(which python)"
echo "Python version: $(python --version)"
"""

    # For resume mode, copy code at runtime into code/{SLURM_JOB_ID}/
    if resume and source_code_dir:
        script += f"""
# ========================================
# Code Copy for Resume Mode
# ========================================
CODE_DIR="{dump_dir}/code/${{SLURM_JOB_ID}}"
echo "Copying code to $CODE_DIR..."
mkdir -p "$CODE_DIR"

# Use rsync to copy code (faster and handles excludes better)
rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \\
    --exclude='.venv' --exclude='venv' --exclude='*.egg-info' --exclude='.eggs' \\
    --exclude='dist' --exclude='build' --exclude='.pytest_cache' --exclude='.mypy_cache' \\
    --exclude='.ruff_cache' --exclude='node_modules' --exclude='*.log' \\
    "{source_code_dir}/" "$CODE_DIR/"

echo "✓ Code copied to $CODE_DIR"
echo "  Python files: $(find $CODE_DIR -name '*.py' | wc -l)"

cd "$CODE_DIR"
"""
    else:
        script += f"""
# Change to code directory
cd {code_dir}
"""

    # Multi-node: Start Ray cluster using srun with --overlap
    if nodes > 1:
        script += f"""
# ========================================
# Ray Cluster Setup (multi-node)
# ========================================
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
HEAD_NODE=${{NODES[0]}}
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w $HEAD_NODE hostname -I | awk '{{print $1}}')
HEAD_PORT=6379
CPUS_PER_NODE={cpus_per_node}

echo "Head node: $HEAD_NODE"
echo "Head IP: $HEAD_IP"
echo "Worker nodes: ${{NODES[@]:1}}"

# Create a cleanup function
cleanup() {{
    echo "Cleaning up Ray cluster..."
    ray stop 2>/dev/null
    # Kill any remaining ray processes
    pkill -f "ray::" 2>/dev/null || true
}}
trap cleanup EXIT

# Start Ray head on first node (we're already on head node)
echo "Starting Ray head on $HEAD_NODE..."
ray start --head --port=$HEAD_PORT --num-cpus=$CPUS_PER_NODE

# Wait for head to be ready
sleep 5

# Start Ray workers on remaining nodes using srun
# Each worker runs in blocking mode in background
for node in "${{NODES[@]:1}}"; do
    echo "Starting Ray worker on $node..."
    srun --nodes=1 --ntasks=1 -w $node --overlap \\
        bash -c "source /storage/home/winnieyangwn/miniforge3/etc/profile.d/conda.sh && conda activate rllm && \\
unset VIRTUAL_ENV && unset PYTHONPATH && ray start --address=$HEAD_IP:$HEAD_PORT --num-cpus=$CPUS_PER_NODE --block" &
done

# Wait for workers to join
sleep 15

echo "Ray cluster ready. Checking status..."
ray status

# Set Ray address for eval script
export RAY_ADDRESS=$HEAD_IP:$HEAD_PORT

# Run evaluation
python {runtime_code_dir}/examples/mlebench/{eval_script} \\
    --config {config_path} \\
    --output-dir {dump_dir}/results \\
    --samples {samples} \\
    {task_args}

EXIT_CODE=$?

# Cleanup is handled by trap
"""
    else:
        # Single-node: Ray starts locally in eval_ray.py
        script += f"""
# Run evaluation (Ray starts locally on this node)
python {runtime_code_dir}/examples/mlebench/{eval_script} \\
    --config {config_path} \\
    --output-dir {dump_dir}/results \\
    --samples {samples} \\
    {task_args}

EXIT_CODE=$?
"""

    script += """
echo "========================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
"""
    return script


def main():
    parser = argparse.ArgumentParser(
        description="SLURM Launcher for MLE-bench Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single-node (Ray starts locally, 64 parallel tasks)
    python launch.py --config configs/gpt5.yaml --name exp_001 --task mlsp-2013-birds

    # Multi-node Ray cluster (3 nodes, tasks distributed automatically)
    python launch.py --config configs/gpt5.yaml --name exp_002 --task mlsp-2013-birds \\
        --nodes 3 --samples 192

    # Multiple tasks
    python launch.py --config configs/gpt5.yaml --name exp_003 --tasks mlsp-2013-birds,spooky-author-identification

    # Custom samples and time limit
    python launch.py --config configs/gpt5.yaml --name exp_003 --task mlsp-2013-birds --samples 32 --time 12:00:00

    # Dry run (shows sbatch script without submitting)
    python launch.py --config configs/gpt5.yaml --name exp_004 --task mlsp-2013-birds --dry-run
        """,
    )

    # Required arguments
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--name", type=str, required=True, help="Experiment name (used in job name and dump dir)")

    # Task specification (optional - can come from config)
    task_group = parser.add_mutually_exclusive_group(required=False)
    task_group.add_argument("--task", type=str, help="Single task ID (default: from config)")
    task_group.add_argument("--tasks", type=str, help="Comma-separated list of task IDs (default: from config)")

    # Optional arguments
    parser.add_argument("--samples", type=int, default=64, help="Samples per prompt (default: 64)")
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition (default: from config or h200)")
    parser.add_argument("--qos", type=str, default=None, help="SLURM QoS (default: from config or h200_coding_shared)")
    parser.add_argument("--account", type=str, default=None, help="SLURM account (default: from config or aira_ws2)")
    parser.add_argument("--time", type=str, default=None, help="Time limit HH:MM:SS (default: from config or 168:00:00)")
    parser.add_argument("--nodes", type=int, default=None, help="Number of nodes (default: from config or 1)")
    parser.add_argument("--dump-dir", type=str, default=str(DEFAULT_DUMP_DIR), help="Base dump directory")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch script without submitting")
    parser.add_argument("--no-copy-code", action="store_true", help="Skip copying code (use existing rllm install)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing dump directory (reuses code/configs, skips completed samples)")

    # Ray mode arguments
    parser.add_argument("--ray", action="store_true", default=True, help="Use Ray for task distribution (default: True)")
    parser.add_argument("--no-ray", action="store_false", dest="ray", help="Disable Ray, use single-threaded eval.py")
    parser.add_argument("--cpus-per-node", type=int, default=None, help="CPUs to register per Ray node (default: from config or 192)")
    parser.add_argument("--gpus-per-node", type=int, default=None, help="GPUs per node (default: from config or 8)")

    args = parser.parse_args()

    # Load config to get SLURM defaults
    config_path = Path(args.config)
    if not config_path.is_absolute():
        if not config_path.exists():
            config_path = RLLM_ROOT / "examples/mlebench" / args.config

    if config_path.exists():
        cfg = OmegaConf.load(config_path)

        # Handle Hydra-style defaults: merge base configs
        # OmegaConf doesn't resolve Hydra defaults, so we do it manually
        if "defaults" in cfg:
            defaults = cfg.get("defaults", [])
            merged_cfg = OmegaConf.create({})
            for default in defaults:
                if isinstance(default, str):
                    base_name = default
                elif isinstance(default, dict):
                    # Handle dict format like {override: value}
                    base_name = list(default.values())[0] if default else None
                else:
                    base_name = None

                if base_name:
                    base_path = config_path.parent / f"{base_name}.yaml"
                    if base_path.exists():
                        base_cfg = OmegaConf.load(base_path)
                        merged_cfg = OmegaConf.merge(merged_cfg, base_cfg)

            # Remove defaults key and merge the main config on top
            cfg_without_defaults = OmegaConf.create({k: v for k, v in cfg.items() if k != "defaults"})
            cfg = OmegaConf.merge(merged_cfg, cfg_without_defaults)

        slurm_cfg = cfg.get("slurm", {})
    else:
        slurm_cfg = {}

    # Apply config defaults (CLI args override config)
    args.partition = args.partition or slurm_cfg.get("partition", "h200")
    args.qos = args.qos or slurm_cfg.get("qos", "h200_coding_shared")
    args.account = args.account or slurm_cfg.get("account", "aira_ws2")
    args.time = args.time or slurm_cfg.get("time", "168:00:00")
    args.nodes = args.nodes or slurm_cfg.get("nodes", 1)
    args.gpus_per_node = args.gpus_per_node or slurm_cfg.get("gpus_per_node", 8)
    args.cpus_per_node = args.cpus_per_node or slurm_cfg.get("cpus_per_node", 192)

    # Get task from config if not specified via CLI
    eval_cfg = cfg.get("eval", {}) if config_path.exists() else {}
    args.task = args.task or eval_cfg.get("task")
    args.tasks = args.tasks or eval_cfg.get("tasks")

    # Validate task is specified somewhere
    if not args.task and not args.tasks:
        print("Error: Must specify task via --task, --tasks, or in config (eval.task / eval.tasks)")
        sys.exit(1)

    # config_path was already resolved above when loading SLURM defaults
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Create dump directory
    dump_dir = create_dump_directory(args.name, Path(args.dump_dir))

    # Check resume mode
    code_dir = dump_dir / "code"
    dump_configs_dir = dump_dir / "configs"

    # Parse task IDs for progress checking
    task_ids = []
    if args.task:
        task_ids = [args.task]
    elif args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]

    if args.resume:
        # Resume mode: validate existing directory structure
        if not dump_dir.exists():
            print(f"Error: Cannot resume - dump directory does not exist: {dump_dir}")
            sys.exit(1)
        if not dump_configs_dir.exists():
            print(f"Error: Cannot resume - configs directory not found: {dump_configs_dir}")
            sys.exit(1)

        print(f"Resuming from existing dump directory: {dump_dir}")
        dump_config_path = dump_configs_dir / config_path.name

        # Check progress for each task
        print("\nChecking progress...")
        progress = check_resume_progress(dump_dir, task_ids, args.samples)

        total_completed = 0
        total_remaining = 0
        for task_id, stats in progress.items():
            total_completed += stats["completed"]
            total_remaining += stats["remaining"]
            status = "✓ DONE" if stats["remaining"] == 0 else f"{stats['remaining']} remaining"
            print(f"  {task_id}: {stats['completed']}/{stats['total']} completed ({status})")

        print(f"\nTotal: {total_completed} completed, {total_remaining} remaining")

        if total_remaining == 0:
            print("\n✓ All rollouts already completed! Nothing to do.")
            sys.exit(0)

        # For resume mode, code will be copied at runtime into code/{SLURM_JOB_ID}/
        # We pass RLLM_ROOT as the source to copy from
        source_code_dir = RLLM_ROOT
        # code_dir is used as a base path but actual dir will be code/{job_id}
    else:
        # Check if dump directory already exists with code
        if code_dir.exists():
            print(f"Error: Dump directory already exists: {dump_dir}")
            print(f"       Code directory found: {code_dir}")
            print("\nOptions:")
            print("  1. Resume from existing: add --resume flag")
            print(f"  2. Start fresh: rm -rf {dump_dir}")
            sys.exit(1)

        print(f"Created dump directory: {dump_dir}")
        source_code_dir = None

        # Copy code (unless --no-copy-code)
        if args.no_copy_code:
            code_dir = RLLM_ROOT
            print("Using existing rllm installation (--no-copy-code)")
        else:
            code_dir = copy_code(dump_dir)

        # Copy configs directory to dump directory (needed for defaults: [base] to work)
        source_configs_dir = config_path.parent
        if dump_configs_dir.exists():
            print(f"Warning: Removing existing configs directory: {dump_configs_dir}")
            shutil.rmtree(dump_configs_dir)
        shutil.copytree(source_configs_dir, dump_configs_dir)
        dump_config_path = dump_configs_dir / config_path.name

        # Also copy base.yaml from root configs if it exists and wasn't already copied
        root_configs_dir = RLLM_ROOT / "examples" / "mlebench" / "configs"
        base_yaml = root_configs_dir / "base.yaml"
        if base_yaml.exists() and not (dump_configs_dir / "base.yaml").exists():
            shutil.copy(base_yaml, dump_configs_dir / "base.yaml")
            print("  + Copied base.yaml for config inheritance")

        print(f"Copied configs to: {dump_configs_dir}")

    # Create logs directory
    logs_dir = dump_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Generate sbatch script (single job, even for multi-node)
    sbatch_content = generate_sbatch_script(
        name=args.name,
        config_path=dump_config_path,
        dump_dir=dump_dir,
        code_dir=code_dir,
        task=args.task,
        tasks=args.tasks,
        samples=args.samples,
        partition=args.partition,
        qos=args.qos,
        account=args.account,
        time=args.time,
        nodes=args.nodes,
        ray=args.ray,
        cpus_per_node=args.cpus_per_node,
        gpus_per_node=args.gpus_per_node,
        resume=args.resume,
        source_code_dir=source_code_dir if args.resume else None,
    )

    # Write sbatch script
    sbatch_path = dump_dir / "run.sbatch"
    sbatch_path.write_text(sbatch_content)
    print(f"Generated sbatch script: {sbatch_path}")

    # Dry run or submit
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - sbatch script content:")
        print("=" * 60)
        print(sbatch_content)
        print("=" * 60)
        print(f"\nTo submit manually:\n  sbatch {sbatch_path}")
    else:
        print("\nSubmitting job...")
        result = subprocess.run(
            ["sbatch", str(sbatch_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            job_id = output.split()[-1] if output else "unknown"
            print(f"✓ {output}")
            print(f"\nDump directory: {dump_dir}")
            print(f"Monitor with: tail -f {logs_dir}/{job_id}.out")
        else:
            print("✗ Failed to submit job")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            sys.exit(1)


if __name__ == "__main__":
    main()
