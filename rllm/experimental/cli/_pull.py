"""HuggingFace dataset download and catalog loading utilities."""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


def load_dataset_catalog() -> dict:
    """Load the datasets.json catalog from the registry directory.

    Returns:
        dict: The full catalog with 'version' and 'datasets' keys.
    """
    catalog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "registry", "datasets.json")
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def load_agent_catalog() -> dict:
    """Load the agents.json catalog from the registry directory.

    Returns:
        dict: The full catalog with 'version' and 'agents' keys.
    """
    catalog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "registry", "agents.json")
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def pull_dataset(name: str, catalog_entry: dict) -> None:
    """Download a dataset from HuggingFace and register it locally.

    Args:
        name: Dataset name (e.g., 'gsm8k').
        catalog_entry: Entry from datasets.json with 'source', 'splits', etc.
    """
    from datasets import load_dataset

    from rllm.data import DatasetRegistry

    source = catalog_entry["source"]
    splits = catalog_entry.get("splits", ["train", "test"])

    logger.info(f"Pulling dataset '{name}' from {source}...")

    for split in splits:
        try:
            hf_dataset = load_dataset(source, split=split)
            DatasetRegistry.register_dataset(
                name=name,
                data=hf_dataset,
                split=split,
                source=source,
                description=catalog_entry.get("description", ""),
                category=catalog_entry.get("category", ""),
            )
            num_examples = len(hf_dataset)
            logger.info(f"  Registered {name}/{split} ({num_examples} examples)")
        except Exception as e:
            logger.warning(f"  Failed to pull {name}/{split}: {e}")
            raise
