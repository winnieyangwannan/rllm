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

def compute_step_score(trajectory):
    """
    Computes the reward for each step in a given trajectory.

    This function extracts and returns a list of augmented rewards for each step 
    in the trajectory. If the trajectory is empty, it returns an empty list.

    Args:
        trajectory (List[Dict]): A list of step dictionaries, where each step 
                                 contains at least the key "augmented_reward".

    Returns:
        List[float]: A list of rewards corresponding to each step in the trajectory.
                     Returns an empty list if the trajectory is empty.
    """
    if not trajectory:
        return []
    result = [0] * (len(trajectory) - 1)
    result.append(np.sum([d["augmented_reward"] for d in trajectory]))
    return result


def compute_environment_score(trajectory):
    """
    Computes the overall environment score for a given trajectory.

    This function extracts the total trajectory reward from the first step in the trajectory.
    If the trajectory is empty, it returns a default score of 0.

    Args:
        trajectory (List[Dict]): A list of step dictionaries, where the first step 
                                 contains the key "trajectory_reward".

    Returns:
        float: The environment score extracted from the first step. Returns 0 if the 
               trajectory is empty.
    """
    return trajectory[0]["trajectory_reward"] if trajectory else 0


def convert_observation_to_string(env_id, obs, with_system_prompt=False):
    """
    Converts an observation into a format suitable for LLM based on the environment type. 

    Args:
        env_id (str): The identifier of the environment.
        obs (Any): The raw observation data from the environment.
        with_system_prompt (bool, optional): Whether to include a system prompt 
                                             in the formatted observation. Default is False.

    Returns:
        Any: The processed observation in the appropriate format for the environment.

    Raises:
        ValueError: If the provided `env_id` is not recognized.

    """
    if env_id.startswith("browsergym/miniwob"):
        return convert_miniwob_observation(obs, with_system_prompt)
    raise ValueError(f"Unknown environment: {env_id}")


def convert_miniwob_observation(obs, with_system_prompt=False):
    """Convert MiniWoB observation to a readable string for LLMs."""
    from rllm.models.web_agent import WebAgent
    # Currently converts using a dummy WebAgent and reuses related methods
    dummy_agent = WebAgent()
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
