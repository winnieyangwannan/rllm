"""Tests for rllm setup CLI command."""

import os
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from rllm.experimental.cli.main import cli
from rllm.experimental.eval.config import load_config, save_config, RllmConfig


@pytest.fixture
def tmp_rllm_home(monkeypatch, tmp_path):
    """Redirect RLLM_HOME to a temp directory."""
    rllm_home = str(tmp_path / ".rllm")
    monkeypatch.setenv("RLLM_HOME", rllm_home)
    return rllm_home


@pytest.fixture
def runner():
    return CliRunner()


def test_setup_fresh(runner, tmp_rllm_home):
    """Setup with no existing config should prompt and save."""
    # Numbered fallback: 1 = openai, API key, 1 = gpt-4o-mini
    result = runner.invoke(cli, ["setup"], input="1\nsk-testkey123\n1\n")
    assert result.exit_code == 0
    assert "Configuration saved" in result.output

    config = load_config()
    assert config.provider == "openai"
    assert config.api_key == "sk-testkey123"
    assert config.model == "gpt-4o-mini"


def test_setup_select_different_model(runner, tmp_rllm_home):
    """Setup should allow selecting a different model from the list."""
    # 1 = openai, API key, 2 = gpt-4o
    result = runner.invoke(cli, ["setup"], input="1\nsk-testkey\n2\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.model == "gpt-4o"


def test_setup_custom_model(runner, tmp_rllm_home):
    """Setup should allow entering a custom model name."""
    from rllm.experimental.eval.config import PROVIDER_MODELS
    other_num = len(PROVIDER_MODELS["openai"]) + 1  # "Other (enter manually)"

    result = runner.invoke(cli, ["setup"], input=f"1\nsk-testkey\n{other_num}\nmy-custom-model\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.model == "my-custom-model"


def test_setup_shows_existing(runner, tmp_rllm_home):
    """Setup should show existing config when re-running."""
    save_config(RllmConfig(provider="openai", api_key="sk-oldkey1234", model="gpt-4o"))

    # 1 = openai, y = keep key, 1 = gpt-4o-mini
    result = runner.invoke(cli, ["setup"], input="1\ny\n1\n")
    assert result.exit_code == 0
    assert "current config" in result.output
    assert "****1234" in result.output


def test_setup_keeps_existing_key(runner, tmp_rllm_home):
    """Setup should keep existing key when user confirms."""
    save_config(RllmConfig(provider="openai", api_key="sk-existing", model="gpt-4o"))

    # 1 = openai, y = keep key, 2 = gpt-4o
    result = runner.invoke(cli, ["setup"], input="1\ny\n2\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.api_key == "sk-existing"


def test_setup_replace_key(runner, tmp_rllm_home):
    """Setup should allow replacing the API key."""
    save_config(RllmConfig(provider="openai", api_key="sk-old", model="gpt-4o"))

    # 1 = openai, n = replace key, new key, 1 = gpt-4o-mini
    result = runner.invoke(cli, ["setup"], input="1\nn\nsk-newkey\n1\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.api_key == "sk-newkey"


def test_setup_default_selection(runner, tmp_rllm_home):
    """Pressing enter should use the default (pre-selected) choice."""
    # Just press enter for provider (default 1) and model (default 1)
    result = runner.invoke(cli, ["setup"], input="\nsk-testkey\n\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"


def test_setup_invalid_choice_reprompts(runner, tmp_rllm_home):
    """Invalid number should reprompt, not crash."""
    # 99 is invalid, then 1 is valid; same for model
    result = runner.invoke(cli, ["setup"], input="99\n1\nsk-testkey\n99\n1\n")
    assert result.exit_code == 0
    assert "Please enter a number" in result.output

    config = load_config()
    assert config.provider == "openai"


def test_setup_with_tty_uses_terminal_menu(tmp_rllm_home):
    """When TTY is available and simple-term-menu installed, TerminalMenu is used."""
    mock_menu_instance = MagicMock()
    mock_menu_instance.show.return_value = 0
    mock_menu_cls = MagicMock(return_value=mock_menu_instance)

    with patch("rllm.experimental.cli.setup._has_tty", return_value=True), \
         patch("rllm.experimental.cli.setup._get_terminal_menu", return_value=mock_menu_cls):
        runner = CliRunner()
        result = runner.invoke(cli, ["setup"], input="sk-testkey\n")

    assert result.exit_code == 0
    assert mock_menu_cls.call_count == 2  # provider + model
