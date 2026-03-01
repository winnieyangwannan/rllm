"""Tests for rllm eval config (load/save/validate)."""

import json
import os
import stat

import pytest

from rllm.experimental.eval.config import (
    DEFAULT_MODELS,
    SUPPORTED_PROVIDERS,
    RllmConfig,
    load_config,
    save_config,
)


@pytest.fixture
def tmp_rllm_home(monkeypatch, tmp_path):
    """Redirect RLLM_HOME to a temp directory."""
    rllm_home = str(tmp_path / ".rllm")
    monkeypatch.setenv("RLLM_HOME", rllm_home)
    return rllm_home


class TestRllmConfig:
    def test_empty_config_not_configured(self):
        config = RllmConfig()
        assert not config.is_configured()

    def test_partial_config_not_configured(self):
        config = RllmConfig(provider="openai", api_key="sk-test")
        assert not config.is_configured()

    def test_full_config_is_configured(self):
        config = RllmConfig(provider="openai", api_key="sk-test", model="gpt-4o")
        assert config.is_configured()

    def test_validate_empty(self):
        config = RllmConfig()
        errors = config.validate()
        assert len(errors) == 3
        assert any("provider" in e for e in errors)
        assert any("api_key" in e for e in errors)
        assert any("model" in e for e in errors)

    def test_validate_unsupported_provider(self):
        config = RllmConfig(provider="unsupported", api_key="key", model="m")
        errors = config.validate()
        assert len(errors) == 1
        assert "unsupported" in errors[0]

    def test_validate_valid(self):
        config = RllmConfig(provider="openai", api_key="sk-test", model="gpt-4o")
        assert config.validate() == []


class TestLoadSaveConfig:
    def test_load_missing_file(self, tmp_rllm_home):
        config = load_config()
        assert not config.is_configured()

    def test_save_and_load(self, tmp_rllm_home):
        original = RllmConfig(provider="openai", api_key="sk-secret", model="gpt-4o-mini")
        path = save_config(original)
        assert os.path.exists(path)

        loaded = load_config()
        assert loaded.provider == "openai"
        assert loaded.api_key == "sk-secret"
        assert loaded.model == "gpt-4o-mini"
        assert loaded.is_configured()

    def test_save_creates_directories(self, tmp_rllm_home):
        config = RllmConfig(provider="openai", api_key="sk-test", model="gpt-4o")
        path = save_config(config)
        assert os.path.isdir(os.path.dirname(path))

    def test_save_sets_permissions(self, tmp_rllm_home):
        config = RllmConfig(provider="openai", api_key="sk-test", model="gpt-4o")
        path = save_config(config)
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600

    def test_load_corrupt_file(self, tmp_rllm_home):
        path = os.path.join(tmp_rllm_home, "config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("not valid json{{{")
        config = load_config()
        assert not config.is_configured()

    def test_save_overwrites(self, tmp_rllm_home):
        save_config(RllmConfig(provider="openai", api_key="key1", model="m1"))
        save_config(RllmConfig(provider="openai", api_key="key2", model="m2"))
        loaded = load_config()
        assert loaded.api_key == "key2"
        assert loaded.model == "m2"


class TestConstants:
    def test_openai_in_supported_providers(self):
        assert "openai" in SUPPORTED_PROVIDERS

    def test_default_models_has_openai(self):
        assert "openai" in DEFAULT_MODELS
