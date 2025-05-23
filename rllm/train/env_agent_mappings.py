# from rllm.environments import BrowserGym, FrozenLakeEnv, ToolEnvironment, SingleTurnEnvironment, SWEEnv, CompetitionCodingEnv

# from rllm.agents import WebAgent, FrozenLakeAgent, ToolAgent, SWEAgent, MathAgent, CompetitionCodingAgent
from rllm.environments.code.competition_coding import CompetitionCodingEnv
from rllm.agents.code_agent import CompetitionCodingAgent

ENV_CLASS_MAPPING = {
    # 'browsergym': BrowserGym,
    # 'frozenlake': FrozenLakeEnv,
    # 'tool': ToolEnvironment,
    # 'math': SingleTurnEnvironment,
    # 'code': SingleTurnEnvironment,
    # 'swe': SWEEnv,
    'competition_coding': CompetitionCodingEnv,
}

AGENT_CLASS_MAPPING = {
    # 'webagent': WebAgent,
    # 'frozenlakeagent': FrozenLakeAgent,
    # 'tool_agent': ToolAgent,
    # 'sweagent': SWEAgent,
    # 'math_agent': MathAgent,
    'code_agent': CompetitionCodingAgent,
} 

def setup_environment(config):
    # if config.env.name == 'browsergym':
    #     if config.env.subtask == 'miniwob':
    #         import os
    #         import importlib
    #         import browsergym.miniwob
    #         importlib.reload(browsergym.miniwob)
    #         os.environ["MINIWOB_URL"] = config.env.miniwob_url
    #         return
    # elif config.env.name in ['frozenlake', 'swe', 'math', 'code', 'tool', 'competition_coding']:
    #     return
    # raise ValueError(f"Environment subtask not supported, env: {config.env.name}, subtask: {config.env.subtask == 'miniwob'}")
    pass