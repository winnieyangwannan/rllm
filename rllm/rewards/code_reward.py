
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

#from rllm.rewards.code_utils.code_contests import run_test as code_contests_run_test
from rllm.rewards.code_utils.livecodebench import run_test as lcb_run_test
from rllm.rewards.code_utils.codeforces import run_test as codeforces_run_test
#from rllm.rewards.code_utils.swebench import swebench_check_correctness
from rllm.rewards.code_utils.taco import run_test as taco_run_test
from rllm.rewards.code_utils.firejail_exec import code_exec_firejail as code_exec
from rllm.rewards.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType


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

def check_correctness(tests: Union[List[Dict[str, str]], Dict[str, List[str]]], code: str, test_fn, timeout_per_test: int = 5, max_tests: int = 1e9) -> bool:
    """
    Check if generated code passes all test cases within a timeout period.

    Args:
        tests: Test cases in either list of dictionaries or dictionary of lists format
        code: Generated code to test
        test_fn: Function to run tests
        timeout: Maximum execution time in seconds before killing process

    Returns:
        bool: True if all tests pass, False otherwise

    Raises:
        AssertionError: If test results list is empty
    """
    manager = Manager()
    test_results = manager.list()
    def evaluate_code(tests, generation, debug, test_results, test_fn):
        """Helper function to run tests in separate process."""
        try:
            test_results.append(test_fn(tests, test=generation, debug=debug))
        except Exception as e:
            print(f"Error in evaluate_code: {e}")

    if isinstance(tests, list):
        total_tests = len(tests)
        if total_tests > max_tests:
            # Randomly select at most 15 test cases
            selected_indices = random.sample(range(total_tests), max_tests)
            tests = [tests[i] for i in selected_indices]
        num_tests = len(tests)
    else:
        total_tests = len(tests['inputs'])
        if total_tests > max_tests:
            # Randomly select at most 15 test cases
            selected_indices = random.sample(range(total_tests), max_tests)
            # Create a new dict with only the selected test cases
            selected_tests = {
                'inputs': [tests['inputs'][i] for i in selected_indices],
                'outputs': [tests['outputs'][i] for i in selected_indices]
            }
            tests = selected_tests
        num_tests = len(tests['inputs'])
    
    timeout = timeout_per_test * num_tests
    
    process = multiprocessing.Process(
        target=evaluate_code,
        args=(tests, code, False, test_results, test_fn)
    )
    process.start()
    process.join(timeout=timeout + 1)

    if process.is_alive():
        process.kill()
    test_results = test_results[:]
    if len(test_results) == 0:
        return False
    #assert len(test_results) == 1, f"Expected exactly one test result, but got {test_results}"
    test_results = test_results[0]
    test_results = [r==True for r in test_results]
    return all(test_results)


def postprocess_lcb_sample(sample):
    sample_inputs = [sample['input'] for sample in sample]
    sample_outputs = [sample['output'] for sample in sample]
    
    sample_dict = {
        'inputs': sample_inputs,
        'outputs': sample_outputs,
    }
    
    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None, f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}"
        # Fill in the blank
        sample_dict['fn_name'] = fn_name
    
    sample = {
        'input_output': json.dumps(sample_dict),
    }
    return sample

#https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems
def verify_check_correctess(tests, code):
    if isinstance(tests, str):
        try:
            tests =  ast.literal_eval(tests)
            assert isinstance(tests, dict)
        except (ValueError, SyntaxError) as e:
            print(f"run_tests app/taco, Error parsing string: {e}")
            return False
    
    def synthesize_std_code(raw_code):
        normal_import_lines = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
        filtered_lines = [line for line in raw_code.splitlines() if "threading.stack_size" not in line]
        filtered_lines = [line for line in filtered_lines if "#! /usr/bin/env python" not in line]
        raw_code = '\n'.join(filtered_lines)
        sol = "" 
        tmp_test = raw_code.split("\n")
        # define the code line type, 1 for import lines, 2 for import * lines with indent, 0 for normal codes
        code_types = [] 

        for x in tmp_test:
            if 'import *' in x:
                code_types.append(2)
            elif x.startswith("from ") or x.startswith("import "):
                code_types.append(1) 
            else:
                code_types.append(0)
        
        started = False
        special_import_lines = [i.lstrip('\t') for idx, i in enumerate(tmp_test) if code_types[idx]==2]
        special_import_lines = '\n'.join(special_import_lines)

        for idx, i in enumerate(tmp_test):
            code_type = code_types[idx]
            if code_type == 0 and not started:
                sol += normal_import_lines
                sol += special_import_lines
                sol += "\nstdin = sys.stdin\nstdout = sys.stdout\n"
                sol += "def code():\n"
                sol += f"\t{i}\n"
                started = True
            else:
                if code_type < 2:
                    if started:
                        sol += '\t'
                    sol += f"{i}\n"

        sol += "code()\n"
        return sol

    def remove_trailing_spaces(output):
        lines = output.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        cleaned_output = '\n'.join(cleaned_lines)
        return cleaned_output
    code = synthesize_std_code(code)
    for test in tests:
        input = test['input']
        output = test['output']
        succ, exec_output = code_exec(code, input)
        clean_output = remove_trailing_spaces(output)
        clean_exec_output = remove_trailing_spaces(exec_output)
        if not ( succ and  ( exec_output.strip() == output.strip() or clean_exec_output.strip() == clean_output.strip() )):
            return False 
    return True
    

def lcb_check_correctness_v2(sample, generation, timeout=6, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    assert len(sample) >= 1, "Sample must contain at least one test case"
    sample = postprocess_lcb_sample(sample)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()


    def _temp_run(sample, generation, debug, result, metadata_list, timeout):
        res, metadata = lcb_run_test(sample, test=generation, debug=debug, timeout=timeout)
        result.append(res)
        metadata_list.append(metadata)

    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    )
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    if not result:
        return False
    print(result[0], metadata_list)
    # Check if all elements in result[0] are True
    return all(x == True for x in result[0])


def leetcode_check_correctness(tests: List[Dict[str, str]], code: str) -> bool:
     """
     Check if generated code passes all LeetCode test cases.
    
     Args:
          tests: List of test cases, each containing input/output pairs
          code: Generated code to test
          timeout: Maximum execution time in seconds before killing process
          runtime_debug: Whether to print debug info during test execution
    
     Returns:
          bool: True if all tests pass and result list exists, False otherwise
     """
     succ, output = code_exec(code + '\n' + tests["functional"])
     if not succ:
         print(f"Error in code execution: {output}")
     return succ

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
        tests = metadata
        if tests is None:
            print("No tests found in metadata")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        model_code = extract_code_from_model(model_response)
        if model_code is None:
            #print("No code found in model response")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Tests: List[Dictionary] - Codeforces, LiveCodeBench
        # Tests: Dictionary[Lists] - CodeContests, Taco/Apps
        is_correct = False
        if dataset_name in ["taco", "apps", "code_contests"]:
            test_fn = taco_run_test
        elif dataset_name == "codeforces":
            test_fn = codeforces_run_test
        
        if dataset_name == "leetcode":
            is_correct = leetcode_check_correctness(tests, model_code)
        elif dataset_name == "livecodebench":
            is_correct = lcb_check_correctness_v2(tests, model_code, debug=False)
        elif dataset_name == "verify":
            is_correct = verify_check_correctess(tests, model_code)
        else:
            is_correct = check_correctness(tests, model_code, test_fn)

        total_time = time.time() - total_start_time
        # print(f"Total reward function execution time: {total_time:.2f} seconds")

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
    True
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
