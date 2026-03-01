"""``rllm setup`` — interactive configuration for provider, API key, and model."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from rllm.experimental.eval.config import (
    DEFAULT_MODELS,
    PROVIDER_MODELS,
    SUPPORTED_PROVIDERS,
    RllmConfig,
    load_config,
    save_config,
)

theme = Theme({"option": "cyan", "option.selected": "bold cyan", "label": "dim", "success": "bold green", "error": "bold red", "key": "yellow"})
console = Console(theme=theme)


def _mask_key(key: str) -> str:
    """Mask an API key for display, showing only the last 4 characters."""
    if len(key) <= 4:
        return "****"
    return "****" + key[-4:]


def _has_tty() -> bool:
    """Check if we have a real terminal for interactive menus."""
    if not sys.stdin.isatty():
        return False
    try:
        with open("/dev/tty"):
            return True
    except OSError:
        return False


def _get_terminal_menu():
    """Lazy-import TerminalMenu. Returns the class or None if unavailable."""
    try:
        from simple_term_menu import TerminalMenu
        return TerminalMenu
    except ImportError:
        return None


def _select_from_menu(title: str, choices: list[str], cursor: int = 0) -> int | None:
    """Show an interactive menu if possible, otherwise fall back to numbered prompt."""
    TerminalMenu = _get_terminal_menu() if _has_tty() else None

    if TerminalMenu is not None:
        console.print(f"  [label]{title}[/]")
        menu = TerminalMenu(
            choices,
            cursor_index=cursor,
            menu_cursor_style=("fg_cyan", "bold"),
            menu_highlight_style=("fg_cyan", "bold"),
        )
        return menu.show()

    # Fallback: numbered list
    console.print(f"  [label]{title}[/]")
    for i, choice in enumerate(choices):
        if i == cursor:
            console.print(f"    [option.selected]> {i + 1}) {choice}[/]")
        else:
            console.print(f"      {i + 1}) [option]{choice}[/]")
    while True:
        raw = Prompt.ask(f"  Enter choice [dim](1-{len(choices)})[/]", default=str(cursor + 1), console=console)
        try:
            idx = int(raw.strip()) - 1
            if 0 <= idx < len(choices):
                return idx
        except ValueError:
            pass
        console.print(f"  [error]Please enter a number between 1 and {len(choices)}.[/]")


def _select_provider(existing: RllmConfig) -> str:
    """Interactive provider selection."""
    choices = list(SUPPORTED_PROVIDERS)
    cursor = 0
    if existing.provider in choices:
        cursor = choices.index(existing.provider)

    idx = _select_from_menu("Provider", choices, cursor)
    if idx is None:
        console.print("\n  [dim]Aborted.[/]")
        raise SystemExit(1)
    return choices[idx]


def _select_model(provider: str, existing: RllmConfig) -> str:
    """Interactive model selection with option to enter a custom model."""
    models = PROVIDER_MODELS.get(provider, [])
    choices = list(models) + ["Other (enter manually)"]

    cursor = 0
    if existing.model in models:
        cursor = models.index(existing.model)
    elif existing.model:
        cursor = len(models)

    idx = _select_from_menu("Model", choices, cursor)
    if idx is None:
        console.print("\n  [dim]Aborted.[/]")
        raise SystemExit(1)

    if idx == len(models):
        default = existing.model if existing.model and existing.model not in models else ""
        model = Prompt.ask("  Enter model name", default=default or None, console=console).strip()
        if not model:
            console.print("  [error]Model is required.[/]")
            raise SystemExit(1)
    else:
        model = choices[idx]

    return model


@click.command("setup")
def setup_cmd():
    """Configure rLLM with your provider, API key, and default model."""
    existing = load_config()

    console.print()
    console.print(Panel("[bold]rLLM Setup[/]", subtitle="[dim]configure your provider and model[/]", border_style="cyan", expand=False))
    console.print()

    if existing.is_configured():
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="label", width=10)
        table.add_column()
        table.add_row("Provider", existing.provider)
        table.add_row("API key", f"[key]{_mask_key(existing.api_key)}[/]")
        table.add_row("Model", existing.model)
        console.print(Panel(table, title="[dim]current config[/]", border_style="dim", expand=False))
        console.print()

    # Provider
    provider = _select_provider(existing)
    console.print()

    # API key
    if existing.api_key and existing.provider == provider:
        console.print(f"  [label]API key[/]  [key]{_mask_key(existing.api_key)}[/]")
        keep = Confirm.ask("  Keep existing key?", default=True, console=console)
        if keep:
            api_key = existing.api_key
        else:
            api_key = Prompt.ask("  New API key", password=True, console=console).strip()
    else:
        api_key = Prompt.ask("  [label]API key[/]", password=True, console=console).strip()

    if not api_key:
        console.print("  [error]API key is required.[/]")
        raise SystemExit(1)
    console.print()

    # Model
    model = _select_model(provider, existing)
    console.print()

    config = RllmConfig(provider=provider, api_key=api_key, model=model)
    errors = config.validate()
    if errors:
        for err in errors:
            console.print(f"  [error]Error: {err}[/]")
        raise SystemExit(1)

    path = save_config(config)

    # Summary
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="label", width=10)
    table.add_column()
    table.add_row("Provider", f"[bold]{provider}[/]")
    table.add_row("API key", f"[key]{_mask_key(api_key)}[/]")
    table.add_row("Model", f"[bold]{model}[/]")
    table.add_row("Saved to", f"[dim]{path}[/]")
    console.print(Panel(table, title="[success]Configuration saved[/]", border_style="green", expand=False))
    console.print()
