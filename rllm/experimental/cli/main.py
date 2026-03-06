"""rLLM CLI: evaluate any model on any benchmark with one command.

Entry point: ``rllm [dataset|eval|agent|model]``
"""

from __future__ import annotations

import click
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

_BANNER = r"""
       _     _     __  __
  _ __| |   | |   |  \/  |
 | '__| |   | |   | |\/| |
 | |  | |__ | |__ | |  | |
 |_|  |____||____||_|  |_|
"""

_COMMAND_ICONS = {
    "agent": "🤖",
    "dataset": "📦",
    "eval": "📊",
    "init": "🚀",
    "model": "⚙️ ",
    "login": "🔑",
    "train": "🏋️",
}

console = Console()


class _LazyGroup(click.Group):
    """A Click group that lazily imports subcommands on first use.

    Avoids importing heavy modules (torch, litellm, transformers) at CLI
    startup by deferring subcommand module imports until a command is
    actually invoked.
    """

    # (module_path, attr_name, short_help)
    _COMMANDS: dict[str, tuple[str, str, str] | None] = {
        "agent": ("rllm.experimental.cli.agent", "agent", "Manage agent scaffolds."),
        "dataset": ("rllm.experimental.cli.dataset", "dataset", "Manage datasets."),
        "eval": ("rllm.experimental.cli.eval", "eval_cmd", "Evaluate a model on a benchmark dataset."),
        "init": ("rllm.experimental.cli.init", "init_cmd", "Scaffold a new agent project."),
        "model": ("rllm.experimental.cli.model_cmd", "model", "Manage provider and model configuration."),
        "train": ("rllm.experimental.cli.train", "train_cmd", "Train a model on a benchmark dataset using RL."),
        "login": ("rllm.experimental.cli.login", "login_cmd", "Log in to rLLM UI."),
        "setup": None,  # handled inline
    }

    def list_commands(self, ctx):
        # Exclude hidden commands (setup)
        return [name for name in sorted(self._COMMANDS) if name != "setup"]

    def get_command(self, ctx, cmd_name):
        if cmd_name == "setup":
            return setup_alias
        spec = self._COMMANDS.get(cmd_name)
        if spec is None:
            return None
        module_path, attr, _help = spec
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)

    def format_help(self, ctx, formatter):
        """Render a fancy Rich help screen instead of plain Click output."""
        from importlib.metadata import version as pkg_version

        try:
            ver = pkg_version("rllm")
        except Exception:
            ver = "dev"

        # Banner
        banner_text = Text(_BANNER, style="bold cyan")
        console.print(banner_text, highlight=False)

        # Tagline + version
        tagline = Text()
        tagline.append("  Reinforcement Learning for Language Agents", style="dim")
        tagline.append(f"  v{ver}", style="bold green")
        console.print(tagline)
        console.print()

        # Usage
        console.print(Text("  Usage: ", style="bold") + Text("rllm ", style="bold cyan") + Text("[command] [options]", style="dim"))
        console.print()

        # Commands table
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
            padding=(0, 2),
            expand=False,
        )
        table.add_column("Command", style="bold cyan", min_width=12)
        table.add_column("Description", style="dim")

        for name in self.list_commands(ctx):
            spec = self._COMMANDS.get(name)
            if spec is None:
                continue
            _mod, _attr, short_help = spec
            icon = _COMMAND_ICONS.get(name, " ")
            table.add_row(f" {icon} {name}", short_help)

        console.print(table)
        console.print()

        # Footer hints
        console.print(Text("  Options:", style="bold"))
        console.print(Text("    --version  ", style="cyan") + Text("Show the version and exit.", style="dim"))
        console.print(Text("    --help     ", style="cyan") + Text("Show this message and exit.", style="dim"))
        console.print()
        console.print(Text("  Run ", style="dim") + Text("rllm <command> --help", style="bold cyan") + Text(" for more information on a command.", style="dim"))
        console.print()

    def format_commands(self, ctx, formatter):
        """Write the subcommand list without importing the subcommand modules."""
        commands = []
        for name in self.list_commands(ctx):
            spec = self._COMMANDS.get(name)
            if spec is None:
                continue
            _mod, _attr, short_help = spec
            commands.append((name, short_help))

        if commands:
            # Match Click's default column width calculation
            limit = formatter.width - 6 - max(len(name) for name, _ in commands)
            rows = []
            for name, help_text in commands:
                rows.append((name, click.utils.make_str(help_text)[:limit] if limit > 0 else ""))
            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.group(cls=_LazyGroup, invoke_without_command=True)
@click.version_option(package_name="rllm")
@click.pass_context
def cli(ctx):
    """rLLM: Reinforcement Learning for Language Agents."""
    if ctx.invoked_subcommand is None:
        formatter = ctx.make_formatter()
        cli.format_help(ctx, formatter)  # renders via Rich console


@cli.command("setup", hidden=True)
@click.pass_context
def setup_alias(ctx):
    """[deprecated] Use ``rllm model setup`` instead."""
    click.echo("Hint: use `rllm model setup` (the `setup` command is deprecated).\n", err=True)
    from rllm.experimental.cli.model_cmd import model_setup

    ctx.invoke(model_setup)


if __name__ == "__main__":
    cli()
