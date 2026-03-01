"""Agent management CLI commands.

``rllm agent [list|info]``
"""

from __future__ import annotations

import click

from rllm.experimental.cli._display import format_table
from rllm.experimental.cli._pull import load_agent_catalog, load_dataset_catalog


@click.group()
def agent():
    """Manage agent scaffolds."""


@agent.command(name="list")
def list_agents():
    """List registered agent scaffolds."""
    catalog = load_agent_catalog()
    agents = catalog.get("agents", {})

    if not agents:
        click.echo("No agents registered.")
        return

    headers = ["Name", "Module", "Description"]
    rows = []
    for name, info in sorted(agents.items()):
        module_fn = f"{info.get('module', '')}.{info.get('function', '')}"
        rows.append([name, module_fn, info.get("description", "")])

    click.echo(format_table(headers, rows))


@agent.command()
@click.argument("name")
def info(name: str):
    """Show agent details and compatible datasets."""
    agent_catalog = load_agent_catalog()
    agents = agent_catalog.get("agents", {})

    if name not in agents:
        available = ", ".join(sorted(agents.keys()))
        click.echo(f"Error: Agent '{name}' not found. Available: {available}")
        raise SystemExit(1)

    entry = agents[name]
    click.echo(f"\nAgent: {name}")
    click.echo(f"  Description:  {entry.get('description', 'N/A')}")
    click.echo(f"  Module:       {entry.get('module', 'N/A')}")
    click.echo(f"  Function:     {entry.get('function', 'N/A')}")

    # Find compatible datasets
    ds_catalog = load_dataset_catalog()
    compatible = []
    for ds_name, ds_info in ds_catalog.get("datasets", {}).items():
        if ds_info.get("default_agent") == name:
            compatible.append(ds_name)

    if compatible:
        click.echo(f"\n  Compatible datasets: {', '.join(compatible)}")
    click.echo()
