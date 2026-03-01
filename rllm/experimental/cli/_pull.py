"""HuggingFace dataset download and catalog loading utilities."""

from __future__ import annotations

import importlib
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


def _load_transform(transform_path: str):
    """Load a transform function from a 'module:function' import path.

    Args:
        transform_path: Colon-separated import path (e.g., 'rllm.data.transforms:gpqa_diamond_transform').

    Returns:
        The transform function.
    """
    module_path, fn_name = transform_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def _remap_fields(row: dict, field_map: dict) -> dict:
    """Rename fields according to field_map. Preserves unmapped fields.

    Args:
        row: Original row dict.
        field_map: Mapping of source field name -> target field name.

    Returns:
        New dict with renamed fields.
    """
    new_row = {}
    for src_key, dst_key in field_map.items():
        if src_key in row:
            new_row[dst_key] = row[src_key]
    # Keep unmapped fields
    for key, val in row.items():
        if key not in field_map:
            new_row[key] = val
    return new_row


def pull_dataset(name: str, catalog_entry: dict) -> None:
    """Download a dataset from HuggingFace and register it locally.

    Supports optional field_map, hf_config, and transform for data normalization.

    Args:
        name: Dataset name (e.g., 'gsm8k').
        catalog_entry: Entry from datasets.json with 'source', 'splits', etc.
    """
    from datasets import load_dataset

    from rllm.data import DatasetRegistry

    source = catalog_entry["source"]
    splits = catalog_entry.get("splits", ["train", "test"])
    hf_config = catalog_entry.get("hf_config")
    field_map = catalog_entry.get("field_map")
    transform_path = catalog_entry.get("transform")

    # Load transform function if specified
    transform_fn = _load_transform(transform_path) if transform_path else None

    logger.info(f"Pulling dataset '{name}' from {source}...")

    for split in splits:
        try:
            # Build load_dataset kwargs
            load_kwargs: dict = {"path": source, "split": split}
            if hf_config:
                load_kwargs["name"] = hf_config

            hf_dataset = load_dataset(**load_kwargs)

            # Convert to list of dicts for transformation
            if transform_fn or field_map:
                data_list = [dict(row) for row in hf_dataset]

                if transform_fn:
                    data_list = [transform_fn(row) for row in data_list]

                if field_map:
                    data_list = [_remap_fields(row, field_map) for row in data_list]

                register_data = data_list
            else:
                register_data = hf_dataset

            DatasetRegistry.register_dataset(
                name=name,
                data=register_data,
                split=split,
                source=source,
                description=catalog_entry.get("description", ""),
                category=catalog_entry.get("category", ""),
            )
            num_examples = len(register_data)
            logger.info(f"  Registered {name}/{split} ({num_examples} examples)")
        except Exception as e:
            logger.warning(f"  Failed to pull {name}/{split}: {e}")
            raise
