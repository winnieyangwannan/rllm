#!/usr/bin/env python3
"""
Step 2b: Data Mount Verification Test

Quick test to verify data is accessible in AgentBox container.

This test creates a container with the same config as the end-to-end test
and verifies that the data mount is working correctly.

Usage:
    python test_step2b_data_real.py --task mlsp-2013-birds
    python test_step2b_data_real.py --task mlsp-2013-birds --manager-uri h200-137-000-067:42499
"""

import argparse
import sys
import time

# ============================================================================
# CONFIG (same as test_step7_end_to_end.py)
# ============================================================================

MANAGER_URI = "h200-137-000-067:42499"

# MLE-bench data
MLE_BENCH_DATA_DIR = "/checkpoint/maui/shared/cache/dojo/tasks/mlebench"

# Superimage config (matches amaia-collab)
SUPERIMAGE_DIR = "/checkpoint/maui_sft/shared/sif"
SUPERIMAGE_VERSION = "2025-05-02v2"
SUPERIMAGE_OVERLAY = "/checkpoint/fair-maui-hs/hotfix/kniu.2025-09-19.cache.overlay.ext3.img"


def test_data_accessibility(task_id: str, manager_uri: str) -> bool:
    """
    Test that data is accessible in the container.

    Returns True if data is accessible, False otherwise.
    """
    # Import here to avoid import errors when just checking --help
    from agentbox import ContainerConfig

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    print(f"\n{'=' * 60}")
    print(f"DATA ACCESSIBILITY TEST: {task_id}")
    print(f"{'=' * 60}")

    # Build data path for this task
    data_path = f"{MLE_BENCH_DATA_DIR}/{task_id}/prepared/public"
    print(f"\nHost data path: {data_path}")
    print("Container mount: /root/data")
    print(f"Manager URI: {manager_uri}")

    # Build container config with data mount
    container_config = ContainerConfig(
        superimage_directory=SUPERIMAGE_DIR,
        superimage_version=SUPERIMAGE_VERSION,
        container_runtime="apptainer",
        read_only_overlays=[SUPERIMAGE_OVERLAY],
        read_only_binds={data_path: "/root/data"},
        working_dir="/workspace",
        env={"HF_HUB_OFFLINE": "1", "NLTK_DATA": "/root/.nltk_data"},
    )

    print("\n[1/4] Creating sandbox with data mount...")
    start_time = time.time()

    sandbox = AgentBoxSandbox(
        name=f"data-test-{task_id}",
        manager_uri=manager_uri,
        container_config=container_config,
    )
    print(f"      ✓ Sandbox created in {time.time() - start_time:.1f}s")

    try:
        # Test 1: Check if /root/data exists
        print("\n[2/4] Checking if /root/data exists...")
        result = sandbox.exec("test -d /root/data && echo 'EXISTS' || echo 'NOT_EXISTS'", timeout=10)
        if "NOT_EXISTS" in result:
            print("      ✗ /root/data does NOT exist")
            return False
        print("      ✓ /root/data exists")

        # Test 2: List contents
        print("\n[3/4] Listing /root/data contents...")
        result = sandbox.exec("ls -lah /root/data", timeout=30)
        print(f"      Output:\n{_indent(result, 8)}")

        if "total 0" in result or result.strip() == "":
            print("      ✗ /root/data is empty!")
            return False
        print("      ✓ /root/data has contents")

        # Test 3: Check for expected files
        print("\n[4/4] Checking for expected MLE-bench files...")
        result = sandbox.exec("cd /root/data && ls -sh *.csv 2>/dev/null | head -10", timeout=30)
        if result.strip():
            print(f"      CSV files found:\n{_indent(result, 8)}")
            print("      ✓ Data appears valid")
        else:
            # Some tasks may not have CSV files, check for any files
            result = sandbox.exec("cd /root/data && ls | head -10", timeout=30)
            print(f"      Files found:\n{_indent(result, 8)}")
            if result.strip():
                print("      ✓ Data appears valid (no CSV files, but has other files)")
            else:
                print("      ✗ No files found in /root/data")
                return False

        print(f"\n{'=' * 60}")
        print("✓ DATA ACCESSIBILITY TEST PASSED")
        print(f"{'=' * 60}\n")
        return True

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False

    finally:
        print("Cleaning up sandbox...")
        try:
            sandbox.close()
        except Exception:
            pass


def _indent(text: str, spaces: int) -> str:
    """Indent each line of text."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.strip().split("\n"))


def main():
    parser = argparse.ArgumentParser(description="Quick test for data accessibility in AgentBox container")
    parser.add_argument(
        "--task",
        type=str,
        default="mlsp-2013-birds",
        help="MLE-bench task ID (default: mlsp-2013-birds)",
    )
    parser.add_argument(
        "--manager-uri",
        type=str,
        default=MANAGER_URI,
        help=f"AgentBox manager URI (default: {MANAGER_URI})",
    )
    args = parser.parse_args()

    success = test_data_accessibility(args.task, args.manager_uri)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
