"""Tests for rllm dataset CLI commands."""

import json
import os
import tempfile

import pytest
from click.testing import CliRunner

from rllm.experimental.cli.main import cli


@pytest.fixture
def tmp_rllm_home(monkeypatch, tmp_path):
    """Set up a temporary RLLM_HOME directory."""
    rllm_home = str(tmp_path / ".rllm")
    monkeypatch.setenv("RLLM_HOME", rllm_home)
    # Patch DatasetRegistry class variables (both new and legacy paths)
    from rllm.data.dataset import DatasetRegistry
    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", rllm_home)
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", os.path.join(rllm_home, "datasets", "registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", os.path.join(rllm_home, "datasets"))
    # Point legacy paths to nonexistent dir to prevent migration from real data
    legacy_dir = str(tmp_path / "legacy_registry")
    monkeypatch.setattr(DatasetRegistry, "_LEGACY_REGISTRY_DIR", legacy_dir)
    monkeypatch.setattr(DatasetRegistry, "_LEGACY_REGISTRY_FILE", os.path.join(legacy_dir, "dataset_registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_LEGACY_DATASET_DIR", os.path.join(legacy_dir, "datasets"))
    return rllm_home


@pytest.fixture
def runner():
    return CliRunner()


def test_dataset_list_empty(runner, tmp_rllm_home):
    """List with no datasets should show a helpful message."""
    result = runner.invoke(cli, ["dataset", "list"])
    assert result.exit_code == 0
    assert "No datasets pulled" in result.output


def test_dataset_list_all(runner, tmp_rllm_home):
    """List --all should show catalog datasets."""
    result = runner.invoke(cli, ["dataset", "list", "--all"])
    assert result.exit_code == 0
    assert "gsm8k" in result.output
    assert "available" in result.output


def test_dataset_info_catalog_only(runner, tmp_rllm_home):
    """Info for a catalog dataset that's not pulled should show catalog info."""
    result = runner.invoke(cli, ["dataset", "info", "gsm8k"])
    assert result.exit_code == 0
    assert "gsm8k" in result.output
    assert "math" in result.output.lower()
    assert "not pulled" in result.output


def test_dataset_info_not_found(runner, tmp_rllm_home):
    """Info for a non-existent dataset should error."""
    result = runner.invoke(cli, ["dataset", "info", "nonexistent_dataset_xyz"])
    assert result.exit_code == 1


def test_dataset_remove_not_found(runner, tmp_rllm_home):
    """Remove a non-existent dataset should show error."""
    result = runner.invoke(cli, ["dataset", "remove", "nonexistent"])
    assert result.exit_code == 0
    assert "not found" in result.output


def test_dataset_register_and_list(runner, tmp_rllm_home):
    """Register a dataset manually and list it."""
    from rllm.data import DatasetRegistry

    data = [{"question": "1+1", "ground_truth": "2"}]
    DatasetRegistry.register_dataset("test_ds", data, split="test")

    result = runner.invoke(cli, ["dataset", "list"])
    assert result.exit_code == 0
    assert "test_ds" in result.output


def test_dataset_register_inspect_remove(runner, tmp_rllm_home):
    """Full lifecycle: register, inspect, remove."""
    from rllm.data import DatasetRegistry

    data = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)]
    DatasetRegistry.register_dataset("lifecycle_ds", data, split="test")

    # Inspect
    result = runner.invoke(cli, ["dataset", "inspect", "lifecycle_ds", "--split", "test", "-n", "2"])
    assert result.exit_code == 0
    assert "q0" in result.output
    assert "q1" in result.output

    # Remove
    result = runner.invoke(cli, ["dataset", "remove", "lifecycle_ds"])
    assert result.exit_code == 0
    assert "Removed" in result.output

    # Confirm removal
    result = runner.invoke(cli, ["dataset", "list"])
    assert "lifecycle_ds" not in result.output
