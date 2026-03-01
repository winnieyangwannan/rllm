"""``rllm setup`` — deprecated, replaced by ``rllm model setup``.

This module re-exports UI helpers and the setup command for backward
compatibility.  New code should import from ``_ui`` and ``model_cmd``.
"""

from __future__ import annotations

# Re-export UI helpers so existing patch paths in tests keep working
from rllm.experimental.cli._ui import (  # noqa: F401
    console,
    theme,
    _has_tty,
    _get_terminal_menu,
    _mask_key,
    _select_from_menu,
    _select_provider,
    _select_model,
)

from rllm.experimental.cli.model_cmd import model_setup as setup_cmd  # noqa: F401
