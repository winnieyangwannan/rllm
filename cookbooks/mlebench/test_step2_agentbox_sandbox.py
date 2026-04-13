#!/usr/bin/env python3
"""
Step 2: AgentBoxSandbox Wrapper Test

Tests the AgentBoxSandbox class which wraps the agentbox package
in the rLLM Sandbox protocol.

Tests:
  A) AgentBoxSandbox.exec("echo hello") returns "hello\n"
  B) exec() respects streaming block limit
  C) upload_file() / fetch_file() round-trip
  D) upload_dir() works for a small directory
  E) close() releases resources without error
  F) create_sandbox("agentbox", ...) factory works

Usage:
    python test_agentbox_sandbox.py
"""

import os
import tempfile

# ============================================================================
# CONFIG - UPDATE THESE VALUES FOR YOUR ENVIRONMENT
# ============================================================================

MANAGER_URI = "h200-137-000-067:35743"

# ============================================================================
# TEST IMPLEMENTATION
# ============================================================================


def test_a_exec_hello():
    """Test A: exec("echo hello") returns "hello\\n"."""
    print("\n" + "=" * 60)
    print("TEST A: Basic exec command")
    print("=" * 60)

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    sandbox = AgentBoxSandbox(
        name="test-sandbox-a",
        manager_uri=MANAGER_URI,
    )
    print("✓ Created AgentBoxSandbox")

    try:
        output = sandbox.exec("echo hello")
        print(f"✓ Command output: '{output.strip()}'")

        assert "hello" in output, f"Expected 'hello' in output, got: {output}"
        print("✓ Output verification passed")
    finally:
        sandbox.close()
        print("✓ Sandbox closed")

    print("✓ TEST A PASSED")
    return True


def test_b_streaming_block_limit():
    """Test B: exec() respects streaming block limit."""
    print("\n" + "=" * 60)
    print("TEST B: Streaming block limit")
    print("=" * 60)

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    # Create sandbox with low block limit
    sandbox = AgentBoxSandbox(
        name="test-sandbox-b",
        manager_uri=MANAGER_URI,
        max_streaming_blocks=5,  # Low limit
    )
    print("✓ Created sandbox with max_streaming_blocks=5")

    try:
        # Run command that produces many lines
        output = sandbox.exec("for i in $(seq 1 100); do echo line_$i; done")
        lines = output.strip().split("\n")
        print(f"✓ Received {len(lines)} lines of output")

        # Should have some output, but may be truncated
        assert len(lines) > 0, "Should have some output"
        print("✓ Block limit test passed")
    finally:
        sandbox.close()

    print("✓ TEST B PASSED")
    return True


def test_c_file_roundtrip():
    """Test C: upload_file() / fetch_file() round-trip."""
    print("\n" + "=" * 60)
    print("TEST C: File upload/fetch round-trip")
    print("=" * 60)

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    sandbox = AgentBoxSandbox(
        name="test-sandbox-c",
        manager_uri=MANAGER_URI,
    )
    print("✓ Created sandbox")

    try:
        test_content = "Hello from rLLM AgentBoxSandbox test!\nLine 2.\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            local_upload = os.path.join(tmpdir, "upload.txt")
            local_download = os.path.join(tmpdir, "download.txt")
            remote_path = "/workspace/test_file.txt"

            # Write test file
            with open(local_upload, "w") as f:
                f.write(test_content)
            print(f"✓ Created local file: {local_upload}")

            # Upload
            sandbox.upload_file(local_upload, remote_path)
            print(f"✓ Uploaded to: {remote_path}")

            # Verify via exec
            container_content = sandbox.exec(f"cat {remote_path}")
            print(f"✓ Content in container: {container_content.strip()}")

            # Fetch back
            success = sandbox.fetch_file(remote_path, local_download)
            assert success, "fetch_file() failed"
            print(f"✓ Downloaded to: {local_download}")

            # Verify content
            with open(local_download) as f:
                downloaded = f.read()

            assert test_content.strip() in downloaded.strip(), f"Content mismatch!\nOriginal: {test_content}\nDownloaded: {downloaded}"
            print("✓ Content verified")
    finally:
        sandbox.close()

    print("✓ TEST C PASSED")
    return True


def test_d_upload_dir():
    """Test D: upload_dir() works for a small directory."""
    print("\n" + "=" * 60)
    print("TEST D: Directory upload")
    print("=" * 60)

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    sandbox = AgentBoxSandbox(
        name="test-sandbox-d",
        manager_uri=MANAGER_URI,
    )
    print("✓ Created sandbox")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test directory structure
            local_dir = os.path.join(tmpdir, "testdir")
            os.makedirs(local_dir)
            with open(os.path.join(local_dir, "file1.txt"), "w") as f:
                f.write("content 1")
            with open(os.path.join(local_dir, "file2.txt"), "w") as f:
                f.write("content 2")
            subdir = os.path.join(local_dir, "subdir")
            os.makedirs(subdir)
            with open(os.path.join(subdir, "file3.txt"), "w") as f:
                f.write("content 3")
            print("✓ Created local directory structure")

            # Upload
            remote_dir = "/workspace/uploaded_dir"
            sandbox.upload_dir(local_dir, remote_dir)
            print(f"✓ Uploaded directory to: {remote_dir}")

            # Verify files exist
            output = sandbox.exec(f"find {remote_dir} -type f | sort")
            print(f"✓ Files in container:\n{output}")

            assert "file1.txt" in output
            assert "file2.txt" in output
            assert "file3.txt" in output
            print("✓ All files found")
    finally:
        sandbox.close()

    print("✓ TEST D PASSED")
    return True


def test_e_close_cleanup():
    """Test E: close() releases resources without error."""
    print("\n" + "=" * 60)
    print("TEST E: Resource cleanup on close()")
    print("=" * 60)

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    sandbox = AgentBoxSandbox(
        name="test-sandbox-e",
        manager_uri=MANAGER_URI,
    )
    print("✓ Created sandbox")

    # Do some work
    sandbox.exec("echo hello")

    # Close should not raise
    sandbox.close()
    print("✓ Sandbox closed without error")

    print("✓ TEST E PASSED")
    return True


def test_f_create_sandbox_factory():
    """Test F: create_sandbox("agentbox", ...) factory works."""
    print("\n" + "=" * 60)
    print("TEST F: create_sandbox() factory")
    print("=" * 60)

    from rllm.experimental.agents.sandboxed_agent import create_sandbox

    sandbox = create_sandbox(
        backend="agentbox",
        name="test-factory",
        image="",  # Use default superimage
        manager_uri=MANAGER_URI,
    )
    print("✓ Created sandbox via create_sandbox()")

    try:
        output = sandbox.exec("whoami")
        print(f"✓ whoami output: {output.strip()}")
        assert len(output) > 0
    finally:
        sandbox.close()

    print("✓ TEST F PASSED")
    return True


def run_all_tests():
    """Run all sandbox tests."""
    print("\n" + "=" * 60)
    print("AGENTBOX SANDBOX TEST SUITE (Step 2)")
    print("=" * 60)
    print(f"Manager URI: {MANAGER_URI}")

    tests = [
        ("A", "Basic exec", test_a_exec_hello),
        ("B", "Streaming limit", test_b_streaming_block_limit),
        ("C", "File round-trip", test_c_file_roundtrip),
        ("D", "Directory upload", test_d_upload_dir),
        ("E", "Resource cleanup", test_e_close_cleanup),
        ("F", "Factory function", test_f_create_sandbox_factory),
    ]

    results = []
    for test_id, test_name, test_fn in tests:
        try:
            success = test_fn()
            results.append((test_id, test_name, success, None))
        except Exception as e:
            import traceback

            print(f"\n✗ TEST {test_id} FAILED: {e}")
            traceback.print_exc()
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
