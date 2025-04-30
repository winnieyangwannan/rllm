
"""
This module contains the RewardCode class, which evaluates code datasets answers
and assigns rewards based on their correctness on unit tests.
"""
import json
import multiprocessing
import re
import time
from multiprocessing import Manager
from typing import List, Dict, Union
import random
import ast 
from rllm.rewards.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType

def check_correctness(test, solution):
    pass

class RewardGaiaFn(RewardFn):
    """
    Reward function for evaluating code dataset answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the unit tests provided
    """
    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.GAIA, "RewardGaiaFn only supports GAIA problems"

        model_response = input.model_response
        metadata = input.metadata #final answer for this model_response
        if model_response == metadata :
            return RewardOutput(is_correct=True, score=1.0)
        else:
            return RewardOutput(is_correct=False, score=0.0)



def rllm_reward_fn_code(data_source: str, llm_solution: str, ground_truth: Dict, **kwargs):
    """Evaluate gaia solutions against ground truth ansters
    
    This function creates a reward function to evaluate code solutions by pass the test_case from groun_truth. It can optionally use a language model
    for more sophisticated answer validation.

    Args:
        data_source: The source/dataset the problem comes from
        llm_solution: The solution string provided by the language model to evaluate
        ground_truth: final answer for this llm_solution
        enable_llm: Whether to enable language model validation for complex cases (default: False)

    Returns:
        bool: True if the solution matchs the final answer, False otherwise
    """
    reward_config = RewardConfig()
    reward_fn = RewardGaiaFn(reward_config)
    reward_response = reward_fn(
        RewardInput(
            problem=None,
            problem_type=RewardType.GAIA,
            data_source=data_source,
            model_response=llm_solution,
            metadata=ground_truth
        ))
    return reward_response.is_correct