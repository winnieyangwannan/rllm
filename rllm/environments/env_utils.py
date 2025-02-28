import numpy as np

def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory

def add_mc_return(trajectory, gamma = 0.95):
    """
    add trajectory reward to the dict of each interaction using Monte Carlo returns for each step
    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1]))*gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1 )/ gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards*gamma_matrix, axis = 1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory

# TODO: add flag for different reward schemes, augmented_reward, reward, mc_return, trajectory_reward
def compute_trajectory_score(trajectory):
    """
    given a trajectory, return a list of rewards, one value for each step.
    """
    return [d["augmented_reward"] for d in trajectory] if trajectory else []
    # return [d["reward"] for d in trajectory] if trajectory else []

def compute_environment_score(trajectory):
    """
    given a trajectory, return the environment score
    """
    return trajectory[0]["trajectory_reward"] if trajectory else 0

def convert_observation_to_prompt(env, i, obs):
    env_id = env.env_id[i]
    if env_id.startswith("browsergym/miniwob"):
        return convert_miniwob_observation(obs, with_system_prompt=True)
    raise ValueError(f"Unknown environment: {env_id}")

def convert_observation(env, i, obs):
    env_id = env.env_id[i]
    if env_id.startswith("browsergym/miniwob"):
        return convert_miniwob_observation(obs)
    raise ValueError(f"Unknown environment: {env_id}")

# TODO: may have better ways to convert miniwob observation to string representation
def convert_miniwob_observation(obs, with_system_prompt=False):
    """Convert MiniWoB observation to a readable string for LLMs."""
    from rllm.models.web_agent import WebAgent
    # Currently converts using a dummy WebAgent and reuses related methods
    dummy_agent = WebAgent(rollout_engine=None, engine_name=None, tokenizer=None)
    obs = dummy_agent._preproc_obs(obs)
    result = ""

    if with_system_prompt:
        result += dummy_agent._get_system_prompt()

    result += "# Goal\n"

    result += "".join(msg["text"] for msg in dummy_agent._format_open_tabs(
        obs["open_pages_urls"],
        obs["open_pages_titles"], 
        obs["active_page_index"]
    ))

    if dummy_agent.use_axtree:
        result += f"# Current page Accessibility Tree\n\n{obs['axtree_txt']}\n\n"

    if dummy_agent.use_html:
        result += f"# Current page DOM\n\n{obs['pruned_html']}\n\n"

    if dummy_agent.use_screenshot:
        result += "".join(msg["text"] for msg in dummy_agent._format_screenshot(obs["screenshot"]))

    if with_system_prompt:
        # Add action space description
        result += dummy_agent._get_action_space_description()

        # Add next action prompt
        result += "# Next action\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action. MAKE SURE TO WRAP YOU FINAL ACTION in ```action``` YOU MUST PUT IN THIS EXACT STYLE FOR THE ACTION TO BE VALID. The content must be in the same format as shown before in the Action Space. Don't just include the chain-of-thought, place the FINAL ACTION from Action Space in ```action```"

    return result
