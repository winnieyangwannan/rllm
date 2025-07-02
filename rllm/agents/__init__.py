def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None

# Define all agent imports
AGENT_IMPORTS = [
    ("rllm.agents.web_agent", "WebAgent"),
    ("rllm.agents.frozenlake_agent", "FrozenLakeAgent"),
    ("rllm.agents.tool_agent", "ToolAgent"),
    ("rllm.agents.swe_agent", "SWEAgent"),
    ("rllm.agents.math_agent", "MathAgent"),
    ("rllm.agents.code_agent", "CompetitionCodingAgent"),
    ("rllm.agents.webarena_agent", "WebArenaAgent"),
]

__all__ = []

for module_path, class_name in AGENT_IMPORTS:
    imported_class = safe_import(module_path, class_name)
    if imported_class is not None:
        globals()[class_name] = imported_class
        __all__.append(class_name)