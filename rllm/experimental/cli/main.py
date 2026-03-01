"""rLLM CLI: evaluate any model on any benchmark with one command.

Entry point: ``rllm [dataset|eval|agent]``
"""

from __future__ import annotations

import click

from rllm.experimental.cli.agent import agent
from rllm.experimental.cli.dataset import dataset
from rllm.experimental.cli.eval import eval_cmd
from rllm.experimental.cli.setup import setup_cmd


@click.group()
@click.version_option(package_name="rllm")
def cli():
    """rLLM: Reinforcement Learning for Language Agents."""


cli.add_command(dataset)
cli.add_command(eval_cmd, name="eval")
cli.add_command(agent)
cli.add_command(setup_cmd, name="setup")


if __name__ == "__main__":
    cli()
