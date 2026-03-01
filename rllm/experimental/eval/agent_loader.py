"""Agent loader: resolves agent by registry name or import path."""

from __future__ import annotations

import importlib
import json
import os

from rllm.experimental.eval.types import AgentFlow


def _load_agent_catalog() -> dict:
    """Load the agents.json catalog from the registry directory."""
    catalog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "registry", "agents.json")
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def load_agent(name_or_path: str) -> AgentFlow:
    """Load an agent by registry name or import path.

    Returns an AgentFlow instance. The loaded object must have a .run() method.

    Args:
        name_or_path: Either a registry name (e.g., "math") or a colon-separated
            import path (e.g., "my_module:my_agent").

    Returns:
        An AgentFlow instance with a .run() method.

    Raises:
        KeyError: If the agent name is not found in the registry.
        ImportError: If the module cannot be imported.
        AttributeError: If the object is not found in the module.
        TypeError: If the loaded object doesn't have a .run() method.
    """
    if ":" in name_or_path:
        # Custom import path: "my_module:my_agent"
        module_path, attr_name = name_or_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, attr_name)
    else:
        # Registry lookup
        catalog = _load_agent_catalog()
        agents = catalog.get("agents", {})
        if name_or_path not in agents:
            available = ", ".join(sorted(agents.keys()))
            raise KeyError(f"Agent '{name_or_path}' not found in registry. Available: {available}")

        entry = agents[name_or_path]
        module = importlib.import_module(entry["module"])
        obj = getattr(module, entry["function"])

    if not hasattr(obj, "run") or not callable(obj.run):
        raise TypeError(
            f"Agent '{name_or_path}' must be an AgentFlow with a .run() method, "
            f"got {type(obj).__name__}"
        )
    return obj
