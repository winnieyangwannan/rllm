"""Dataset management CLI commands.

``rllm dataset [list|pull|info|inspect|remove]``
"""

from __future__ import annotations

import click

from rllm.experimental.cli._display import format_table
from rllm.experimental.cli._pull import load_dataset_catalog, pull_dataset


@click.group()
def dataset():
    """Manage datasets."""


@dataset.command(name="list")
@click.option("--all", "show_all", is_flag=True, help="Show all available datasets from the catalog including undownloaded ones.")
def list_datasets(show_all: bool):
    """List datasets."""
    from rllm.data import DatasetRegistry

    catalog = load_dataset_catalog()
    catalog_datasets = catalog.get("datasets", {})
    local_names = set(DatasetRegistry.get_dataset_names())

    if show_all:
        headers = ["Name", "Category", "Status", "Description"]
        rows = []
        for name, info in sorted(catalog_datasets.items()):
            status = "pulled" if name in local_names else "available"
            rows.append([name, info.get("category", ""), status, info.get("description", "")])
        # Also show locally registered datasets not in catalog
        for name in sorted(local_names - set(catalog_datasets.keys())):
            ds_info = DatasetRegistry.get_dataset_info(name)
            cat = ds_info.get("metadata", {}).get("category", "") if ds_info else ""
            rows.append([name, cat, "local", ""])
        click.echo(format_table(headers, rows))
    else:
        if not local_names:
            click.echo("No datasets pulled yet. Use 'rllm dataset list --all' to see available datasets.")
            return
        headers = ["Name", "Splits", "Category"]
        rows = []
        for name in sorted(local_names):
            splits = DatasetRegistry.get_dataset_splits(name)
            ds_info = DatasetRegistry.get_dataset_info(name)
            cat = ds_info.get("metadata", {}).get("category", "") if ds_info else ""
            rows.append([name, ", ".join(splits), cat])
        click.echo(format_table(headers, rows))


@dataset.command()
@click.argument("name")
def pull(name: str):
    """Pull a dataset from HuggingFace."""
    catalog = load_dataset_catalog()
    catalog_datasets = catalog.get("datasets", {})

    if name not in catalog_datasets:
        available = ", ".join(sorted(catalog_datasets.keys()))
        click.echo(f"Error: Dataset '{name}' not found in catalog. Available: {available}")
        raise SystemExit(1)

    click.echo(f"Pulling {name} from {catalog_datasets[name]['source']}...")
    pull_dataset(name, catalog_datasets[name])
    click.echo(f"Done. Use 'rllm dataset info {name}' to view details.")


@dataset.command()
@click.argument("name")
def info(name: str):
    """Show dataset metadata and splits."""
    from rllm.data import DatasetRegistry

    # Check local registry first
    ds_info = DatasetRegistry.get_dataset_info(name)

    # Also check catalog
    catalog = load_dataset_catalog()
    catalog_entry = catalog.get("datasets", {}).get(name)

    if not ds_info and not catalog_entry:
        click.echo(f"Error: Dataset '{name}' not found.")
        raise SystemExit(1)

    click.echo(f"\nDataset: {name}")

    if catalog_entry:
        click.echo(f"  Description:    {catalog_entry.get('description', 'N/A')}")
        click.echo(f"  Source:         {catalog_entry.get('source', 'N/A')}")
        click.echo(f"  Category:       {catalog_entry.get('category', 'N/A')}")
        click.echo(f"  Default agent:  {catalog_entry.get('default_agent', 'N/A')}")
        click.echo(f"  Reward fn:      {catalog_entry.get('reward_fn', 'N/A')}")
        click.echo(f"  Eval split:     {catalog_entry.get('eval_split', 'N/A')}")

    if ds_info:
        click.echo("\n  Local splits:")
        for split, split_info in ds_info.get("splits", {}).items():
            num = split_info.get("num_examples", "?")
            fields = split_info.get("fields", [])
            click.echo(f"    {split}: {num} examples")
            if fields:
                click.echo(f"      fields: {', '.join(fields)}")
    else:
        click.echo("\n  Status: not pulled (use 'rllm dataset pull {name}')".format(name=name))

    click.echo()


@dataset.command()
@click.argument("name")
@click.option("--split", default=None, help="Split to inspect (default: first available or eval_split).")
@click.option("-n", "--num-rows", default=3, help="Number of example rows to show.")
def inspect(name: str, split: str | None, num_rows: int):
    """Show sample data rows from a dataset."""
    from rllm.data import DatasetRegistry

    catalog = load_dataset_catalog()
    catalog_entry = catalog.get("datasets", {}).get(name)

    if split is None:
        if catalog_entry:
            split = catalog_entry.get("eval_split", "test")
        else:
            splits = DatasetRegistry.get_dataset_splits(name)
            split = splits[0] if splits else "default"

    ds = DatasetRegistry.load_dataset(name, split)
    if ds is None:
        click.echo(f"Error: Cannot load '{name}' split '{split}'. Try 'rllm dataset pull {name}' first.")
        raise SystemExit(1)

    click.echo(f"\n{name}/{split} — {len(ds)} examples (showing first {min(num_rows, len(ds))})\n")

    for i in range(min(num_rows, len(ds))):
        row = ds[i]
        click.echo(f"--- Example {i} ---")
        for key, value in row.items():
            val_str = str(value)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            click.echo(f"  {key}: {val_str}")
        click.echo()


@dataset.command()
@click.argument("name")
@click.option("--split", default=None, help="Remove only this split (default: remove all).")
def remove(name: str, split: str | None):
    """Remove a local dataset."""
    from rllm.data import DatasetRegistry

    if split:
        ok = DatasetRegistry.remove_dataset_split(name, split)
        if ok:
            click.echo(f"Removed {name}/{split}.")
        else:
            click.echo(f"Error: {name}/{split} not found.")
    else:
        ok = DatasetRegistry.remove_dataset(name)
        if ok:
            click.echo(f"Removed {name}.")
        else:
            click.echo(f"Error: Dataset '{name}' not found.")
