"""AgentBox container sandbox backend.

Wraps the standalone `agentbox` package to implement the rLLM Sandbox protocol.
Used for MLE-bench and other GPU-intensive sandboxed agent workloads.

Requires the `agentbox` package: pip install agentbox
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default config values (FAIR cluster)
DEFAULT_SUPERIMAGE_DIR = "/checkpoint/maui_sft/shared/sif"
DEFAULT_SUPERIMAGE_VERSION = "2025-05-02v2"
DEFAULT_SUPERIMAGE_OVERLAY = "/checkpoint/fair-maui-hs/hotfix/kniu.2025-09-19.cache.overlay.ext3.img"


class AgentBoxSandbox:
    """Sandbox protocol implementation backed by the agentbox package.

    Manages the full lifecycle: Manager → Machine → Container.
    The container runs on a GPU node with optional data bind-mounts.

    Example:
        sandbox = AgentBoxSandbox(
            name="my-task",
            manager_uri="h200-137-000-067:35743",
        )
        output = sandbox.exec("echo hello")
        sandbox.close()
    """

    def __init__(
        self,
        name: str,
        manager_uri: str,
        container_config: Any | None = None,
        # Convenience params (used if container_config is None):
        image: str = "",
        superimage_directory: str = "",
        superimage_version: str = "",
        read_only_overlays: list[str] | None = None,
        read_only_binds: dict[str, str] | None = None,
        working_dir: str = "/workspace",
        env: dict[str, str] | None = None,
        container_runtime: str = "apptainer",
        max_streaming_blocks: int = 100,
        **kwargs,  # Absorb extra kwargs for compatibility
    ):
        # Lazy import to avoid requiring agentbox for non-agentbox users
        from agentbox import AgentBoxManager, ContainerConfig

        self.name = name
        self._max_streaming_blocks = max_streaming_blocks

        # Connect to AgentBox manager
        self._manager = AgentBoxManager(manager_uri)
        self._machine = self._manager.start_machine(name=name, blocking=True)
        logger.info("AgentBoxSandbox %s: started machine", name)

        # Build container config if not provided
        if container_config is None:
            # Use defaults for superimage if not specified
            if not superimage_directory:
                superimage_directory = DEFAULT_SUPERIMAGE_DIR
            if not superimage_version:
                superimage_version = DEFAULT_SUPERIMAGE_VERSION
            if read_only_overlays is None:
                read_only_overlays = [DEFAULT_SUPERIMAGE_OVERLAY]

            container_config = ContainerConfig(
                image_handle=image if image else None,
                superimage_directory=superimage_directory,
                superimage_version=superimage_version,
                read_only_overlays=read_only_overlays,
                read_only_binds=read_only_binds or {},
                working_dir=working_dir,
                container_runtime=container_runtime,
                env=env or {},
            )

        self._container = self._machine.start_container(config=container_config, name=name)
        logger.info("AgentBoxSandbox %s: started container", name)

    # ── Sandbox protocol methods ─────────────────────────────────────────────

    def exec(self, command: str, timeout: float | None = None) -> str:
        """Execute command via streaming shell.

        Buffers up to max_streaming_blocks output blocks, then truncates.
        On 'Killed' in output, queries dmesg for OOM context.
        """
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
        """Upload a file from host to container."""
        from agentbox import HOST2CONTAINER

        self._container.copy_file(local_path, remote_path, HOST2CONTAINER)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload directory by tarring, uploading, and extracting."""
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            tar_path = f.name
        try:
            subprocess.run(
                ["tar", "-czf", tar_path, "-C", local_path, "."],
                check=True,
            )
            remote_tar = f"{remote_path}/__upload.tar.gz"
            self.exec(f"mkdir -p {remote_path}")
            self.upload_file(tar_path, remote_tar)
            self.exec(f"cd {remote_path} && tar -xzf __upload.tar.gz && rm __upload.tar.gz")
        finally:
            os.unlink(tar_path)

    def start_agent_process(self, command: str, port: int) -> None:
        """Start a long-running process in the background.

        Note: AgentBox containers are typically short-lived per-task containers,
        so this method may have limited use. Raises NotImplementedError if called.
        """
        raise NotImplementedError("AgentBox sandbox does not support long-running processes. Use exec() for individual commands instead.")

    def get_endpoint(self, port: int) -> tuple[str, dict[str, str]]:
        """Return endpoint URL for a port in the container.

        Note: AgentBox containers typically don't expose ports externally.
        """
        raise NotImplementedError("AgentBox sandbox does not support port endpoints. Commands must be executed via exec().")

    def close(self) -> None:
        """Release container and machine resources."""
        try:
            self._machine.free_container(self._container)
            logger.info("AgentBoxSandbox %s: freed container", self.name)
        except Exception:
            logger.exception("Failed to free container %s", self.name)

        try:
            self._manager.free_machine(self._machine)
            logger.info("AgentBoxSandbox %s: freed machine", self.name)
        except Exception:
            logger.exception("Failed to free machine %s", self.name)

    # ── Extra methods for MLE-bench evaluation ───────────────────────────────

    def fetch_file(self, remote_path: str, local_path: str) -> bool:
        """Download a file from the container to the host.

        Returns True on success, False on failure.
        """
        from agentbox import CONTAINER2HOST

        try:
            self._container.copy_file(remote_path, local_path, CONTAINER2HOST)
            return True
        except Exception:
            logger.warning("Failed to fetch file %s from container %s", remote_path, self.name)
            return False

    def list_directory(self, path: str) -> list:
        """List files in a container directory."""
        try:
            return self._container.list_directory(path)
        except Exception:
            return []


def create_agentbox_sandbox(
    name: str,
    manager_uri: str,
    **kwargs,
) -> AgentBoxSandbox:
    """Factory function for creating an AgentBoxSandbox."""
    return AgentBoxSandbox(name=name, manager_uri=manager_uri, **kwargs)
