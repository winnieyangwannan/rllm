"""rLLM configuration: persistent provider/model settings for eval.

Stores configuration in ``~/.rllm/config.json`` (or ``$RLLM_HOME/config.json``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

SUPPORTED_PROVIDERS = ["openai"]
DEFAULT_MODELS: dict[str, str] = {"openai": "gpt-4o-mini"}
PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "o3-mini", "o4-mini"],
}


def _rllm_home() -> str:
    return os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm"))


def _config_path() -> str:
    return os.path.join(_rllm_home(), "config.json")


@dataclass
class RllmConfig:
    """User-level rLLM configuration."""

    provider: str = ""
    api_key: str = ""
    model: str = ""

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

    Returns an empty ``RllmConfig`` if the file is missing or corrupt.
    """
    path = _config_path()
    if not os.path.exists(path):
        return RllmConfig()
    try:
        with open(path) as f:
            data = json.load(f)
        return RllmConfig(
            provider=data.get("provider", ""),
            api_key=data.get("api_key", ""),
            model=data.get("model", ""),
        )
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
    data = {"provider": config.provider, "api_key": config.api_key, "model": config.model}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.chmod(path, 0o600)
    return path
