"""Tests for rllm model CLI command group."""

import json
import os

import pytest
from click.testing import CliRunner

from rllm.experimental.cli.main import cli
from rllm.experimental.eval.config import RllmConfig, load_config, save_config


@pytest.fixture
def tmp_rllm_home(monkeypatch, tmp_path):
    """Redirect RLLM_HOME to a temp directory."""
    rllm_home = str(tmp_path / ".rllm")
    monkeypatch.setenv("RLLM_HOME", rllm_home)
    return rllm_home


@pytest.fixture
def runner():
    return CliRunner()


# --- model setup ---


def test_model_setup_fresh(runner, tmp_rllm_home):
    """First-time setup: provider -> key -> model."""
    # 1 = openai, key, 1 = gpt-5-nano
    result = runner.invoke(cli, ["model", "setup"], input="1\nsk-test123\n1\n")
    assert result.exit_code == 0
    assert "Configuration saved" in result.output

    config = load_config()
    assert config.provider == "openai"
    assert config.api_key == "sk-test123"
    assert config.model == "gpt-5-nano"
    assert config.api_keys == {"openai": "sk-test123"}


def test_model_setup_already_configured_no_swap(runner, tmp_rllm_home):
    """If already configured and user says no, config is unchanged."""
    save_config(RllmConfig(provider="openai", model="gpt-5-mini", api_keys={"openai": "sk-orig"}))

    result = runner.invoke(cli, ["model", "setup"], input="n\n")
    assert result.exit_code == 0
    assert "No changes made" in result.output

    config = load_config()
    assert config.provider == "openai"
    assert config.model == "gpt-5-mini"
    assert config.api_key == "sk-orig"


def test_model_setup_already_configured_swap(runner, tmp_rllm_home):
    """If already configured and user says yes, delegates to swap flow."""
    save_config(RllmConfig(provider="openai", model="gpt-5-mini", api_keys={"openai": "sk-orig"}))

    # y = swap, 2 = anthropic, new key, 1 = first model
    result = runner.invoke(cli, ["model", "setup"], input="y\n2\nsk-ant-key\n1\n")
    assert result.exit_code == 0
    assert "Configuration saved" in result.output

    config = load_config()
    assert config.provider == "anthropic"
    assert config.api_key == "sk-ant-key"
    assert config.api_keys["openai"] == "sk-orig"  # preserved


# --- model swap ---


def test_model_swap_same_provider_new_model(runner, tmp_rllm_home):
    """Swap model within same provider — key is preserved without prompting."""
    save_config(RllmConfig(provider="openai", model="gpt-5-nano", api_keys={"openai": "sk-keep"}))

    # 1 = openai, n = don't change key, 2 = gpt-5-mini
    result = runner.invoke(cli, ["model", "swap"], input="1\nn\n2\n")
    assert result.exit_code == 0
    assert "Configuration saved" in result.output

    config = load_config()
    assert config.provider == "openai"
    assert config.model == "gpt-5-mini"
    assert config.api_key == "sk-keep"


def test_model_swap_new_provider_prompts_key(runner, tmp_rllm_home):
    """Swap to a provider with no stored key — prompts for one."""
    save_config(RllmConfig(provider="openai", model="gpt-5-mini", api_keys={"openai": "sk-oai"}))

    # 2 = anthropic (no key), key, 1 = first model
    result = runner.invoke(cli, ["model", "swap"], input="2\nsk-ant-new\n1\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.provider == "anthropic"
    assert config.api_key == "sk-ant-new"
    assert config.api_keys["openai"] == "sk-oai"  # old key preserved


def test_model_swap_known_provider_no_key_prompt(runner, tmp_rllm_home):
    """Swap to a provider that already has a stored key — no key prompt."""
    save_config(
        RllmConfig(
            provider="openai",
            model="gpt-5-mini",
            api_keys={"openai": "sk-oai", "anthropic": "sk-ant-stored"},
        )
    )

    # 2 = anthropic (has stored key), n = don't change key, 1 = first model
    result = runner.invoke(cli, ["model", "swap"], input="2\nn\n1\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.provider == "anthropic"
    assert config.api_key == "sk-ant-stored"


def test_model_swap_requires_setup(runner, tmp_rllm_home):
    """Swap should fail if not yet configured."""
    result = runner.invoke(cli, ["model", "swap"])
    assert result.exit_code != 0
    assert "Not configured" in result.output


# --- model show ---


def test_model_show(runner, tmp_rllm_home):
    """Show should display current config."""
    save_config(RllmConfig(provider="openai", model="gpt-5-mini", api_keys={"openai": "sk-showme1234"}))

    result = runner.invoke(cli, ["model", "show"])
    assert result.exit_code == 0
    assert "openai" in result.output
    assert "gpt-5-mini" in result.output
    assert "****1234" in result.output


def test_model_show_not_configured(runner, tmp_rllm_home):
    """Show should indicate not configured when no config exists."""
    result = runner.invoke(cli, ["model", "show"])
    assert result.exit_code == 0
    assert "Not configured" in result.output


# --- backward compatibility ---


def test_config_backward_compat(tmp_rllm_home):
    """Old format with 'api_key' string should load correctly."""
    config_path = os.path.join(tmp_rllm_home, "config.json")
    os.makedirs(tmp_rllm_home, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump({"provider": "openai", "api_key": "sk-legacy", "model": "gpt-5-mini"}, f)

    config = load_config()
    assert config.provider == "openai"
    assert config.api_key == "sk-legacy"
    assert config.api_keys == {"openai": "sk-legacy"}
    assert config.model == "gpt-5-mini"
    assert config.is_configured()
