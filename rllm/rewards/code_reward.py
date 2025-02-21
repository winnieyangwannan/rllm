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


from rllm.rewards.code_utils.code_contests import run_test as code_contests_run_test
from rllm.rewards.code_utils.livecodebench import run_test as lcb_run_test
from rllm.rewards.code_utils.codeforces import run_test as codeforces_run_test
from rllm.rewards.code_utils.swebench import swebench_check_correctness
from rllm.rewards.code_utils.taco import run_test as taco_run_test
from rllm.rewards.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType


def all_true(result):
    if isinstance(result, list):  # Check if it's a list
        return all(all_true(item) for item in result)  # Recursively check all elements
    return result is True

def extract_code_from_model(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()

def check_correctness(tests: Union[List[Dict[str, str]], Dict[str, List[str]]], code: str, test_fn, timeout=300):
    manager = Manager()
    test_results = manager.list()
    def evaluate_code(tests, generation, debug, test_results, test_fn):
        try:
            test_results.append(test_fn(tests, test=generation, debug=debug))
        except Exception as e:
            print(f"Error in evaluate_code: {e}")   
    
    p = multiprocessing.Process(target=evaluate_code, args=(tests, code, True, test_results, test_fn))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    test_results = test_results[:]
    return bool(test_results and all_true(test_results[0]))

def lcb_check_correctness(tests, code: str, timeout=30, runtime_debug=False, is_extracted=False):
    result_list = lcb_run_test(tests, code, timeout, runtime_debug, is_extracted)
    details = [r[0] for r in result_list]
    all_passed = all(details)
    result = ""
    if result_list and all_passed:
        result = "passed"
    return result == "passed"
    

class RewardCodeFn(RewardFn):
    """
    Reward function for evaluating code dataset answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the unit tests provided
    """
    def __call__(self, input: RewardInput) -> RewardOutput:
        total_start_time = time.time()

        assert input.problem_type == RewardType.CODE, \
            "Invalid problem type: expected 'CODE', but got '{}'".format(input.problem_type)

        model_response= input.model_response
        metadata= input.metadata
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                print(f"Unable to parse metadata: {e}")
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        dataset_name = input.data_source
        tests = metadata.get("tests", None)
        if tests is None:
            print("No tests found in metadata")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        model_code = extract_code_from_model(model_response)
        print(f"model_code: {model_code}")
        if model_code is None:
            print("No code found in model response")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Tests: List[Dictionary] - Codeforces, LiveCodeBench
        # Tests: Dictionary[Lists] - CodeContests, Taco/Apps
        # Tests: str - TACO/Apps -> Diciotnary[Lists]
        is_correct = False
        if dataset_name in ["taco", "apps"]:
            test_fn = taco_run_test
        elif dataset_name == "code_contests":
            test_fn = code_contests_run_test
        elif dataset_name == "codeforces":
            test_fn = codeforces_run_test
        
        if dataset_name != "livecodebench":
            is_correct = check_correctness(tests, model_code, test_fn)
        else:
            is_extracted = not metadata["tests"][0].get("testtype") == "stdin"
            is_correct = lcb_check_correctness(tests, model_code, is_extracted=is_extracted)
        
        total_time = time.time() - total_start_time
        print(f"Total reward function execution time: {total_time:.2f} seconds")

        if is_correct:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def rllm_reward_fn_code(data_source: str, llm_solution: str, ground_truth: Dict, **kwargs):
    """Evaluate code solutions against ground truth ansters
    
    This function creates a reward function to evaluate code solutions by pass the test_case from groun_truth. It can optionally use a language model
    for more sophisticated answer validation.

    Args:
        data_source: The source/dataset the problem comes from
        llm_solution: The solution string provided by the language model to evaluate
        ground_truth: some tests for this llm_solution
        enable_llm: Whether to enable language model validation for complex cases (default: False)

    Returns:
        bool: True if the solution passes all the test_case, False otherwise

    Example:
            model_response = '''
import sys
from itertools import permutations
def main():
    n,m=map(int, input().split()) 
    a=sum(list(map(int, input().split()))) 
    if a+(n-1)*10<=m: 
        print(5) 
    else: 
        print(5)
if __name__ == "__main__":
    main()
'''
    
    print(f"test the code_forces")
    # tests = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 10\n3 2 1", "output": "5" } ] 
    metadata = {
         "tests": tests,
    }
    Truez
    """
    reward_config = RewardConfig()
    reward_fn = RewardCodeFn(reward_config)
    reward_response = reward_fn(
        RewardInput(
            problem=None,
            problem_type=RewardType.CODE,
            data_source=data_source,
            model_response=llm_solution,
            metadata=ground_truth
        ))
    return reward_response.is_correct
