from rllm.environments.browsergym.browsergym import BrowserGym
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.environments import ToolEnvironment#, SingleTurnEnvironment
# from rllm.environments.swe.swe import SWEEnv

from rllm.agents.web_agent import WebAgent
from rllm.agents.frozenlake_agent import FrozenLakeAgent
from rllm.agents.tool_agent import ToolAgent
from rllm.agents.swe_agent import SWEAgent
# from rllm.models.math_agent import MathAgent

ENV_CLASS_MAPPING = {
    'browsergym': BrowserGym,
    'frozenlake': FrozenLakeEnv,
    # 'sweenv': SWEEnv,
    'tool': ToolEnvironment,
    # 'math': SingleTurnEnvironment,
}

AGENT_CLASS_MAPPING = {
    'webagent': WebAgent,
    'frozenlakeagent': FrozenLakeAgent,
    'tool_agent': ToolAgent,
    'sweagent': SWEAgent,
    # 'math_agent': MathAgent,
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