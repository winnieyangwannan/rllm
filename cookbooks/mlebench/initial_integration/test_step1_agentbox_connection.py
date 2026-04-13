#!/usr/bin/env python3
"""
Step 1: AgentBox Raw Connection Test

This script verifies the standalone `agentbox` pip package works end-to-end.
No rLLM imports - pure agentbox testing.

Tests:
  A) Connect to the AgentBox manager and start/stop a machine
  B) Start a container with a basic ContainerConfig and run `echo hello`
  C) Verify streaming shell output works (multi-block output)
  D) Test copy_file in both directions (host→container, container→host)
  E) Verify free_container and free_machine release resources cleanly

Usage:
    python test_agentbox_connection.py

Before running, update the CONFIG section below with your environment values.
"""

import os
import tempfile
from pathlib import Path

from agentbox import CONTAINER2HOST, HOST2CONTAINER, AgentBoxManager, ContainerConfig

# ============================================================================
# CONFIG - UPDATE THESE VALUES FOR YOUR ENVIRONMENT
# ============================================================================

MANAGER_URI = "h200-137-000-067:35743"  # Your AgentBox manager URI

# Container image configuration
SUPERIMAGE_DIRECTORY = "/checkpoint/maui_sft/shared/sif"
SUPERIMAGE_VERSION = "2025-05-02v2"

# Optional: read-only overlays for cached packages
READ_ONLY_OVERLAYS = [
    "/checkpoint/fair-maui-hs/hotfix/kniu.2025-09-19.cache.overlay.ext3.img",
]

# ============================================================================
# TEST IMPLEMENTATION
# ============================================================================


def test_a_connect_to_manager():
    """Test A: Connect to the AgentBox manager and start/stop a machine."""
    print("\n" + "=" * 60)
    print("TEST A: Connect to AgentBox manager")
    print("=" * 60)

    mgr = AgentBoxManager(MANAGER_URI)
    print(f"✓ Created AgentBoxManager with URI: {MANAGER_URI}")

    machine = mgr.start_machine(name="test-conn-a", blocking=True)
    print(f"✓ Started machine: {machine}")

    mgr.free_machine(machine)
    print("✓ Freed machine")

    print("✓ TEST A PASSED: Manager connection works")
    return True


def test_b_start_container_and_exec():
    """Test B: Start a container with ContainerConfig and run `echo hello`."""
    print("\n" + "=" * 60)
    print("TEST B: Start container and execute command")
    print("=" * 60)

    mgr = AgentBoxManager(MANAGER_URI)
    machine = mgr.start_machine(name="test-conn-b", blocking=True)
    print("✓ Started machine")

    config = ContainerConfig(
        superimage_directory=SUPERIMAGE_DIRECTORY,
        superimage_version=SUPERIMAGE_VERSION,
        container_runtime="apptainer",
        working_dir="/workspace",
        read_only_overlays=READ_ONLY_OVERLAYS,
    )
    print("✓ Created ContainerConfig")

    container = machine.start_container(config=config, name="test-conn-b")
    print(f"✓ Started container: {container}")

    # Execute a simple command
    output_parts = []
    with container.shell(work_dir=Path("/workspace")) as shell:
        for block in shell.execute("echo hello"):
            output_parts.append(block.output)

    output = "".join(output_parts).strip()
    print(f"✓ Command output: '{output}'")

    assert "hello" in output, f"Expected 'hello' in output, got: {output}"
    print("✓ Output verification passed")

    # Cleanup
    machine.free_container(container)
    print("✓ Freed container")

    mgr.free_machine(machine)
    print("✓ Freed machine")

    print("✓ TEST B PASSED: Container exec works")
    return True


def test_c_streaming_shell_output():
    """Test C: Verify streaming shell output works (multi-block output)."""
    print("\n" + "=" * 60)
    print("TEST C: Streaming shell output (multi-block)")
    print("=" * 60)

    mgr = AgentBoxManager(MANAGER_URI)
    machine = mgr.start_machine(name="test-conn-c", blocking=True)

    config = ContainerConfig(
        superimage_directory=SUPERIMAGE_DIRECTORY,
        superimage_version=SUPERIMAGE_VERSION,
        container_runtime="apptainer",
        working_dir="/workspace",
        read_only_overlays=READ_ONLY_OVERLAYS,
    )
    container = machine.start_container(config=config, name="test-conn-c")
    print("✓ Started container")

    # Run a command that produces multiple lines of output
    output_parts = []
    block_count = 0
    with container.shell(work_dir=Path("/workspace")) as shell:
        for block in shell.execute("for i in 1 2 3 4 5; do echo line_$i; done"):
            output_parts.append(block.output)
            block_count += 1

    output = "".join(output_parts)
    print(f"✓ Received {block_count} output block(s)")
    print(f"✓ Combined output:\n{output}")

    # Verify all lines are present
    for i in range(1, 6):
        assert f"line_{i}" in output, f"Missing line_{i} in output"
    print("✓ All expected lines found in output")

    # Cleanup
    machine.free_container(container)
    mgr.free_machine(machine)
    print("✓ Cleanup complete")

    print("✓ TEST C PASSED: Streaming output works")
    return True


def test_d_file_copy_roundtrip():
    """Test D: Test copy_file in both directions (host→container, container→host)."""
    print("\n" + "=" * 60)
    print("TEST D: File copy round-trip")
    print("=" * 60)

    mgr = AgentBoxManager(MANAGER_URI)
    machine = mgr.start_machine(name="test-conn-d", blocking=True)

    config = ContainerConfig(
        superimage_directory=SUPERIMAGE_DIRECTORY,
        superimage_version=SUPERIMAGE_VERSION,
        container_runtime="apptainer",
        working_dir="/workspace",
        read_only_overlays=READ_ONLY_OVERLAYS,
    )
    container = machine.start_container(config=config, name="test-conn-d")
    print("✓ Started container")

    # Create a test file on the host
    test_content = "Hello from host! Testing agentbox file transfer.\nLine 2.\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_upload = os.path.join(tmpdir, "upload_test.txt")
        local_download = os.path.join(tmpdir, "download_test.txt")
        remote_path = "/workspace/test_file.txt"

        # Write test file on host
        with open(local_upload, "w") as f:
            f.write(test_content)
        print(f"✓ Created local test file: {local_upload}")

        # Upload: host → container
        container.copy_file(local_upload, remote_path, HOST2CONTAINER)
        print(f"✓ Uploaded to container: {remote_path}")

        # Verify the file exists in container
        output_parts = []
        with container.shell(work_dir=Path("/workspace")) as shell:
            for block in shell.execute(f"cat {remote_path}"):
                output_parts.append(block.output)
        container_content = "".join(output_parts)
        print(f"✓ File content in container:\n{container_content}")

        # Download: container → host
        container.copy_file(remote_path, local_download, CONTAINER2HOST)
        print(f"✓ Downloaded from container: {local_download}")

        # Verify downloaded content matches original
        with open(local_download) as f:
            downloaded_content = f.read()

        assert test_content in downloaded_content or downloaded_content.strip() == test_content.strip(), f"Content mismatch!\nOriginal: {test_content}\nDownloaded: {downloaded_content}"
        print("✓ File content matches after round-trip")

    # Cleanup
    machine.free_container(container)
    mgr.free_machine(machine)
    print("✓ Cleanup complete")

    print("✓ TEST D PASSED: File copy works in both directions")
    return True


def test_e_resource_cleanup():
    """Test E: Verify free_container and free_machine release resources cleanly."""
    print("\n" + "=" * 60)
    print("TEST E: Resource cleanup")
    print("=" * 60)

    mgr = AgentBoxManager(MANAGER_URI)

    # Start multiple machines and containers
    machines = []
    containers = []

    for i in range(2):
        machine = mgr.start_machine(name=f"test-cleanup-{i}", blocking=True)
        machines.append(machine)
        print(f"✓ Started machine {i}")

        config = ContainerConfig(
            superimage_directory=SUPERIMAGE_DIRECTORY,
            superimage_version=SUPERIMAGE_VERSION,
            container_runtime="apptainer",
            working_dir="/workspace",
            read_only_overlays=READ_ONLY_OVERLAYS,
        )
        container = machine.start_container(config=config, name=f"test-cleanup-{i}")
        containers.append((machine, container))
        print(f"✓ Started container {i}")

    # Free all resources in reverse order
    for i, (machine, container) in enumerate(reversed(containers)):
        machine.free_container(container)
        print(f"✓ Freed container {len(containers) - 1 - i}")

    for i, machine in enumerate(reversed(machines)):
        mgr.free_machine(machine)
        print(f"✓ Freed machine {len(machines) - 1 - i}")

    print("✓ TEST E PASSED: All resources cleaned up without errors")
    return True


def run_all_tests():
    """Run all connection tests."""
    print("\n" + "=" * 60)
    print("AGENTBOX CONNECTION TEST SUITE")
    print("=" * 60)
    print(f"Manager URI: {MANAGER_URI}")
    print(f"Superimage: {SUPERIMAGE_DIRECTORY}/{SUPERIMAGE_VERSION}")

    tests = [
        ("A", "Manager connection", test_a_connect_to_manager),
        ("B", "Container exec", test_b_start_container_and_exec),
        ("C", "Streaming output", test_c_streaming_shell_output),
        ("D", "File copy", test_d_file_copy_roundtrip),
        ("E", "Resource cleanup", test_e_resource_cleanup),
    ]

    results = []
    for test_id, test_name, test_fn in tests:
        try:
            success = test_fn()
            results.append((test_id, test_name, success, None))
        except Exception as e:
            print(f"\n✗ TEST {test_id} FAILED: {e}")
            results.append((test_id, test_name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, _, success, _ in results if success)
    total = len(results)

    for test_id, test_name, success, error in results:
        status = "✓ PASSED" if success else f"✗ FAILED: {error}"
        print(f"  Test {test_id} ({test_name}): {status}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)
