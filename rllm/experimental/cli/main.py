"""rLLM CLI: evaluate any model on any benchmark with one command.

Entry point: ``rllm [dataset|eval|agent|model]``
"""

from __future__ import annotations

import click


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
        "model": ("rllm.experimental.cli.model_cmd", "model", "Manage provider and model configuration."),
        "train": ("rllm.experimental.cli.train", "train_cmd", "Train a model on a benchmark dataset using RL."),
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


@click.group(cls=_LazyGroup)
@click.version_option(package_name="rllm")
def cli():
    """rLLM: Reinforcement Learning for Language Agents."""


@cli.command("setup", hidden=True)
@click.pass_context
def setup_alias(ctx):
    """[deprecated] Use ``rllm model setup`` instead."""
    click.echo("Hint: use `rllm model setup` (the `setup` command is deprecated).\n", err=True)
    from rllm.experimental.cli.model_cmd import model_setup
    ctx.invoke(model_setup)


if __name__ == "__main__":
    cli()
