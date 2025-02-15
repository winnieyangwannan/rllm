"""
This module contains the RewardCode class, which evaluates code datasets answers
and assigns rewards based on their correctness on unit tests.
"""
from typing import List, Union
from rllm.globals import MODEL_NAME_OR_PATH
from rllm.rewards.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
import multiprocessing
import numpy as np
from multiprocessing import Manager
from rllm.rewards.taco.testing_util import run_test as taco_run_test
from rllm.rewards.code_contests.testing_util import run_test as code_contests_run_test
from rllm.rewards.codeforces.testing_util import run_test as codeforces_run_test
from rllm.rewards.swebench.testing_util import swebench_check_correctness

def _temp_run(problem, generation, debug, result, test_fn):
    try:
        result.append(test_fn(problem, test=generation, debug=debug))
    except Exception as e:
        print(f"Error in _temp_run: {e}")

def check_correctness(problem, generation, test_fn):
    TIME_OUT = 300

    manager = Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(problem, generation, False, result, test_fn))
    p.start()
    p.join(timeout=TIME_OUT + 1)
    if p.is_alive():
        p.kill()
    return bool(result and np.all(result[0]))


class RewardCodeFn(RewardFn):

    """
    Reward function for evaluating code dataset answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the unit tests provided
    """
    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.CODE, \
            "Invalid problem type: expected 'CODE', but got '{}'".format(input.problem_type)

        problem = input.problem
        model_response= input.model_response
        metadata= input.metadata
        dataset_name = metadata.get("dataset_flag", None)
        tests = metadata.get("tests", None)
        if tests is None:
            raise ValueError("No tests found in metadata")
        
        if dataset_name == "TACO":#apps/TACO:
            is_correct = check_correctness(metadata, model_response, taco_run_test)
        elif dataset_name == "code_contests":#codetests
            is_correct = check_correctness(metadata, model_response, code_contests_run_test)
        elif dataset_name == "codeforces":#codeforces 
            is_correct = check_correctness(metadata, model_response, codeforces_run_test)
        elif dataset_name == "swebench":#swebench
            resolve_rate = swebench_check_correctness(
                model_response=model_response,
                metadata=metadata
            )

            reward = 2 * resolve_rate - 1

            return RewardOutput(reward=reward, is_correct=reward == 1)
        else:
            raise ValueError("No supported dataset found")
        print(f"Is correct: {is_correct}")
        if is_correct:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def rllm_code_reward_fn(data_source: str, solution_str: str, ground_truth):
    reward_config = RewardConfig()
    reward_fn = RewardCodeFn(reward_config)
    reward_response = reward_fn(
        RewardInput(
            problem=solution_str,
            problem_type=RewardType.CODE,
            model_response=solution_str,
            metadata={
                "dataset_flag": data_source,
                "tests": ground_truth
            }
        ))
    return reward_response.is_correct