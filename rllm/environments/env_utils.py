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

def add_training_reward(trajectory, training_reward):
    """
    add training reward to the dict of each interaction
    """
    for d in trajectory:
        d.update({"training_reward": training_reward})
    return trajectory

def compute_training_score(trajectory):
    """
    Computes the reward for training in a given trajectory.

    Args:
        trajectory (List[Dict]): A list of step dictionaries, where each step 
                                 contains at least the key "augmented_reward".

    Returns:
        float: The training score extracted from the first step. Returns 0 if the 
               trajectory is empty.
    """
    return trajectory[0]["training_reward"] if trajectory else 0


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