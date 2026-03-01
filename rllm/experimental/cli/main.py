"""rLLM CLI: evaluate any model on any benchmark with one command.

Entry point: ``rllm [dataset|eval|agent|model]``
"""

from __future__ import annotations

import click

from rllm.experimental.cli.agent import agent
from rllm.experimental.cli.dataset import dataset
from rllm.experimental.cli.eval import eval_cmd
from rllm.experimental.cli.model_cmd import model, model_setup


@click.group()
@click.version_option(package_name="rllm")
def cli():
    """rLLM: Reinforcement Learning for Language Agents."""


cli.add_command(dataset)
cli.add_command(eval_cmd, name="eval")
cli.add_command(agent)
cli.add_command(model)


@cli.command("setup", hidden=True)
@click.pass_context
def setup_alias(ctx):
    """[deprecated] Use ``rllm model setup`` instead."""
    click.echo("Hint: use `rllm model setup` (the `setup` command is deprecated).\n", err=True)
    ctx.invoke(model_setup)


if __name__ == "__main__":
    cli()
