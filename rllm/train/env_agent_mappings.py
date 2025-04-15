from rllm.environments import BrowserGym, FrozenLakeEnv, ToolEnvironment, SingleTurnEnvironment

from rllm.agents import WebAgent, FrozenLakeAgent, ToolAgent, SWEAgent, MathAgent


ENV_CLASS_MAPPING = {
    'browsergym': BrowserGym,
    'frozenlake': FrozenLakeEnv,
    'tool': ToolEnvironment,
    'math': SingleTurnEnvironment,
}

AGENT_CLASS_MAPPING = {
    'webagent': WebAgent,
    'frozenlakeagent': FrozenLakeAgent,
    'tool_agent': ToolAgent,
    'sweagent': SWEAgent,
    'math_agent': MathAgent,
} 

def setup_environment(config):
    if config.env.name == 'browsergym':
        if config.env.subtask == 'miniwob':
            import os
            import importlib
            import browsergym.miniwob
            importlib.reload(browsergym.miniwob)
            os.environ["MINIWOB_URL"] = config.env.miniwob_url
            return
    elif config.env.name == 'frozenlake':
        return
    elif config.env.name == "sweenv":
        return
    elif config.env.name == 'tool':
        return
    elif config.env.name == 'math':
        return

    raise ValueError(f"Environment subtask not supported, env: {config.env.name}, subtask: {config.env.subtask == 'miniwob'}")