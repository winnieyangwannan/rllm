def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        raise e
        return None

# Import environment classes
ENV_CLASSES = {
    'browsergym': safe_import('rllm.environments.browsergym.browsergym', 'BrowserGym'),
    'frozenlake': safe_import('rllm.environments.frozenlake.frozenlake', 'FrozenLakeEnv'),
    'tool': safe_import('rllm.environments.tools.tool_env', 'ToolEnvironment'),
    'math': safe_import('rllm.environments.base.single_turn_env', 'SingleTurnEnvironment'),
    'code': safe_import('rllm.environments.base.single_turn_env', 'SingleTurnEnvironment'),
    'swe': safe_import('rllm.environments.swe.swe', 'SWEEnv'),
    'competition_coding': safe_import('rllm.environments.code.competition_coding', 'CompetitionCodingEnv'),
    'browsergym_cloud': safe_import('rllm.environments.browsergym.browsergym_cloud', 'BrowserGymCloud'),
}

# Import agent classes
AGENT_CLASSES = {
    'webagent': safe_import('rllm.agents.web_agent', 'WebAgent'),
    'webarenaagent': safe_import('rllm.agents.webarena_agent', 'WebArenaAgent'),
    'frozenlakeagent': safe_import('rllm.agents.frozenlake_agent', 'FrozenLakeAgent'),
    'tool_agent': safe_import('rllm.agents.tool_agent', 'ToolAgent'),
    'sweagent': safe_import('rllm.agents.swe_agent', 'SWEAgent'),
    'math_agent': safe_import('rllm.agents.math_agent', 'MathAgent'),
    'code_agent': safe_import('rllm.agents.code_agent', 'CompetitionCodingAgent'),
}

# Filter out None values for unavailable imports
ENV_CLASS_MAPPING = {k: v for k, v in ENV_CLASSES.items() if v is not None}
AGENT_CLASS_MAPPING = {k: v for k, v in AGENT_CLASSES.items() if v is not None}

def setup_environment(config):        
    if config.env.name == 'browsergym':
        assert hasattr(config.env.env_args, 'subtask'), "subtask must be defined in environment argument for browsergym"
        if config.env.env_args.subtask == 'miniwob':
            import os
            import importlib
            import browsergym.miniwob
            importlib.reload(browsergym.miniwob)
            assert hasattr(config.env.env_args, 'miniwob_url'), "miniwob_url must be defined in environment argument for browsergym miniwob"
            os.environ["MINIWOB_URL"] = config.env.env_args.miniwob_url
            return
        elif config.env.env_args.subtask == 'webarena':
            return
    elif config.env.name in ['frozenlake', 'swe', 'math', 'code', 'tool', 'competition_coding', 'browsergym_cloud', 'custom']:
        return
    raise ValueError(f"Environment subtask not supported, env: {config.env.name}")