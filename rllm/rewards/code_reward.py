"""
This module contains the RewardCode class, which evaluates code datasets answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from rllm.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
from rllm.rewards.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
import copy
import json
import multiprocessing
import os
import random
import re
import numpy as np
from typing import Dict
from typing import List, Union
from multiprocessing import Manager
from rllm.rewards.taco.testing_util import run_test as taco_run_test
from rllm.rewards.code_contests.testing_util import run_test as code_contests_run_test
from rllm.rewards.codeforces.testing_util import run_test as codeforces_run_test
from rllm.rewards.livecodebench.testing_util import unsafe_lcb_runTests
import json

def _temp_run(problem, generation, debug, result, test_fn):
    try:
        result.append(test_fn(problem, test=generation, debug=debug))
    except Exception as e:
        print(f"Error in _temp_run: {e}")

def all_true(result):
    print(result)
    if isinstance(result, list):  # Check if it's a list
        return all(all_true(item) for item in result)  # Recursively check all elements
    return result is True

def extract_code(llm_output):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        llm_output (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    # Regular expression to match code blocks (both with and without language specifiers)
    code_block_pattern = re.findall(r"```(?:\w+)?\n(.*?)```", llm_output, re.DOTALL)

    # If multiple code blocks exist, join them with a newline
    if code_block_pattern:
        return "\n".join(code_block_pattern).strip()
    
    return llm_output  # Return original string if not found


def check_correctness(problem, generation, test_fn):
    TIME_OUT = 300
    cleaned_code = extract_code(generation)
    # print(f"cleaned_code: {cleaned_code}")   

    manager = Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(problem, cleaned_code, False, result, test_fn))
    p.start()
    p.join(timeout=TIME_OUT + 1)
    if p.is_alive():
        p.kill()
    return bool(result and all_true(result[0]))

def lcb_check_correctness(problem, generation, timeout =6,runtime_debug=False, is_extracted=False):
    result_list = unsafe_lcb_runTests(problem, generation, timeout, runtime_debug, is_extracted)
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
        assert input.problem_type == RewardType.CODE, \
            "Invalid problem type: expected 'CODE', but got '{}'".format(input.problem_type)

        problem = input.problem
        model_response= input.model_response
        metadata= input.metadata
        # Check correctness of the generated code
        if metadata.get("input_output") is not None:#apps/TACO:
            is_correct = check_correctness(metadata, model_response, taco_run_test)
        elif metadata.get("public_tests") is not None:#codetests
            is_correct = check_correctness(metadata, model_response, code_contests_run_test)
        elif metadata.get("test_cases") is not None:#codeforces #TODO(xiaoxiang):fix the codeforces_run_test
            is_correct = check_correctness(metadata, model_response, codeforces_run_test)
        elif metadata.get("public_test_cases") is not None:#livecodebench
            is_extrcted = not metadata["public_test_cases"][0].get("testtype") == "stdin"
            is_correct = lcb_check_correctness(metadata, model_response, is_extracted=is_extrcted)
        else:
            raise ValueError("Invalid metadata format")
        if is_correct:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def rllm_reward_fn(solution_str: str, ground_truth: Dict, enable_llm = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardCodeFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.CODE, model_response=solution_str, metadata=ground_truth))
    return reward_response.is_correct


if __name__ == "__main__":
    #test the codetest
    model_response = """
import sys
from itertools import permutations
def main():
    # Read input
    N, M, R = map(int, sys.stdin.readline().split())
    r = list(map(int, sys.stdin.readline().split()))
    A, B, C = [], [], []
    for _ in range(M):
        a, b, c = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
        C.append(c)

    # Initialize distance matrix
    INF = float('inf')
    dist = [[INF for _ in range(N+1)] for _ in range(N+1)]
    for i in range(1, N+1):
        dist[i][i] = 0

    # Set initial distances
    for i in range(M):
        a, b, c = A[i], B[i], C[i]
        dist[a][b] = c
        dist[b][a] = c

    # Floyd-Warshall algorithm
    for k in range(1, N+1):
        for i in range(1, N+1):
            for j in range(1, N+1):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Generate all permutations of R towns
    min_dist = INF
    for perm in permutations(r):
        total = 0
        for i in range(R-1):
            total += dist[perm[i]][perm[i+1]]
        if total < min_dist:
            min_dist = total

    # Output the minimum distance
    print(min_dist)

if __name__ == "__main__":
    main()
    """



    public_tests= {"input": ["3\n4 5\n6 3\n10 2\n"], "output": ["5\n3 4\n4 4 1 2\n"]}
    metadata = {
        "public_tests": public_tests,
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata)
    output = reward(input)
    print(f"codetest output:{output}")

    # test app/taco
    model_response = """
import sys
from itertools import permutations
def main():
    # Read input
    x= map(int, sys.stdin.readline().split())
    print(5)


if __name__ == "__main__":
    main()
    """

    input_output = {"inputs": ["3\n4\n6\n"], "outputs": ["5\n5\n5\n"]}
    metadata = {
        "input_output": input_output,
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata)
    output = reward(input)
    print(f"app/taco output:{output}")

    #test the code_forces
    model_response = """
import sys
from itertools import permutations
def main():
    n,m=map(int, input().split()) 
    a=sum(list(map(int, input().split()))) 
    if a+(n-1)*10<=m: 
        print((m-a)//5) 
    else: 
        print(01)
if __name__ == "__main__":
    main()
    """
    print(f"test the code_forces")
    test_cases = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 20\n2 1 1", "output": "-1" }, { "input": "50 10000\n5 4 10 9 9 6 7 7 7 3 3 7 7 4 7 4 10 10 1 7 10 3 1 4 5 7 2 10 10 10 2 3 4 7 6 1 8 4 7 3 8 8 4 10 1 1 9 2 6 1", "output": "1943" }, { "input": "50 10000\n4 7 15 9 11 12 20 9 14 14 10 13 6 13 14 17 6 8 20 12 10 15 13 17 5 12 13 11 7 5 5 2 3 15 13 7 14 14 19 2 13 14 5 15 3 19 15 16 4 1", "output": "1891" }]
    # test_cases = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 10\n3 2 1", "output": "5" } ] 
    metadata = {
         "test_cases": test_cases,
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata)
    output = reward(input)
    print(f"code_forces output:{output}")

    #livecodebench
    model_response = """
Yes of course!
```python
import json

def main(phone_numbers):
    # 假设输入是一个电话号码的列表，返回重复的电话号码个数
    seen = set()
    duplicates = set()
    for number in phone_numbers:
        if number in seen:
            duplicates.add(number)
        else:
            seen.add(number)
    
    return len(duplicates)+1

if __name__ == "__main__":
    main(input.strip().split())
```
""" 
    #public_test_case = [{"input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n", "output": "YES\nYES\nYES\nNO\nNO\nYES\n", "testtype": "stdin"}]
    public_test_case = [
    {
        'input': '["12345", "530391", "12345"]',
        'output': '2',
        'testtype': 'functional'
    }
    ]
    metadata = {
        "public_test_cases": public_test_case,
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata)
    output = reward(input)
    print(f"livecodebench output:{output}")

    #test the code_forces
    model_response = """
Sorry I can't help with that!
    """
    print(f"test the code_forces")
    test_cases = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 20\n2 1 1", "output": "-1" }, { "input": "50 10000\n5 4 10 9 9 6 7 7 7 3 3 7 7 4 7 4 10 10 1 7 10 3 1 4 5 7 2 10 10 10 2 3 4 7 6 1 8 4 7 3 8 8 4 10 1 1 9 2 6 1", "output": "1943" }, { "input": "50 10000\n4 7 15 9 11 12 20 9 14 14 10 13 6 13 14 17 6 8 20 12 10 15 13 17 5 12 13 11 7 5 5 2 3 15 13 7 14 14 19 2 13 14 5 15 3 19 15 16 4 1", "output": "1891" }]
    # test_cases = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 10\n3 2 1", "output": "5" } ] 
    metadata = {
         "test_cases": test_cases,
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata)
    output = reward(input)
    print(f"code_forces output:{output}")
    