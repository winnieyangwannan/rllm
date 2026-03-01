"""rLLM configuration: persistent provider/model settings for eval.

Stores configuration in ``~/.rllm/config.json`` (or ``$RLLM_HOME/config.json``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

SUPPORTED_PROVIDERS = ["openai", "anthropic", "gemini"]
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5-mini",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-2.5-flash",
}
PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": [
        # GPT-5 family
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
        # GPT-4 family
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-4o",
        # o-series reasoning
        "o3-mini",
        "o3",
        "o3-pro",
        "o4-mini",
    ],
    "anthropic": [
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-6",
    ],
    "gemini": [
        # Gemini 3 family
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-3.1-pro-preview",
        # Gemini 2.5 family
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        # Gemini 2.0
        "gemini-2.0-flash",
    ],
}
PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _rllm_home() -> str:
    return os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm"))


def _config_path() -> str:
    return os.path.join(_rllm_home(), "config.json")


@dataclass
class RllmConfig:
    """User-level rLLM configuration."""

    provider: str = ""
    model: str = ""
    api_keys: dict[str, str] = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        """Return the API key for the active provider."""
        return self.api_keys.get(self.provider, "")

    def is_configured(self) -> bool:
        """Return True if all required fields are set."""
        return bool(self.provider and self.api_key and self.model)

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty if valid)."""
        errors: list[str] = []
        if not self.provider:
            errors.append("provider is required")
        elif self.provider not in SUPPORTED_PROVIDERS:
            errors.append(f"unsupported provider '{self.provider}' (supported: {', '.join(SUPPORTED_PROVIDERS)})")
        if not self.api_key:
            errors.append("api_key is required")
        if not self.model:
            errors.append("model is required")
        return errors


def load_config() -> RllmConfig:
    """Load configuration from ``~/.rllm/config.json``.

    Handles both old format (``{"api_key": "..."}``)) and new format
    (``{"api_keys": {...}}``), migrating transparently.

    Returns an empty ``RllmConfig`` if the file is missing or corrupt.
    """
    path = _config_path()
    if not os.path.exists(path):
        return RllmConfig()
    try:
        with open(path) as f:
            data = json.load(f)
        provider = data.get("provider", "")
        model = data.get("model", "")

        # New format: api_keys dict
        api_keys: dict[str, str] = dict(data.get("api_keys", {}))

        # Backward compat: old format had a single "api_key" field
        if not api_keys and data.get("api_key") and provider:
            api_keys[provider] = data["api_key"]

        return RllmConfig(provider=provider, model=model, api_keys=api_keys)
    except (json.JSONDecodeError, OSError, TypeError):
        return RllmConfig()


def save_config(config: RllmConfig) -> str:
    """Persist configuration to ``~/.rllm/config.json``.

    Creates parent directories as needed and sets file permissions to 0o600
    (owner read/write only) since the file contains API keys.

    Returns the path that was written.
    """
    path = _config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "provider": config.provider,
        "model": config.model,
        "api_keys": dict(config.api_keys),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.chmod(path, 0o600)
    return path
