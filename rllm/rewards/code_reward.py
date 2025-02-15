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
from multiprocessing import Manager
from rllm.rewards.taco.testing_util import run_test as taco_run_test
from rllm.rewards.code_contests.testing_util import run_test as code_contests_run_test
from rllm.rewards.codeforces.testing_util import run_test as codeforces_run_test
import json

def check_correctness(problem, generation, test_fn):
    TIME_OUT = 300
    def _temp_run(problem, generation, debug, result):
        try:
            result.append(test_fn(problem, test=generation, debug=debug))
        except Exception as e:
            print(f"Error in _temp_run: {e}")

    manager = Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(problem, generation, False, result))
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
        if metadata.get("input_output") is not None:#apps/TACO:
        # Check correctness of the generated code
            print(f"this dataset is from apps/taco")
            is_correct = check_correctness(metadata, model_response, taco_run_test)
        elif metadata.get("public_tests") is not None:#codetests
            print(f"this dataset is from code_tests")
            is_correct = check_correctness(metadata, model_response, code_contests_run_test)
        elif metadata.get("test_cases") is not None:#codeforces #TODO(xiaoxiang):fix the codeforces_run_test
            print(f"this dataset is from codeforces")
            is_correct = check_correctness(metadata, model_response, codeforces_run_test)
        else:
            raise ValueError("Invalid metadata format")
        if is_correct:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)


if __name__ == "__main__":
    # #test the codetest
    # model_response = """
    # import sys
    # from itertools import permutations
    # def main():
    #     # Read input
    #     N, M, R = map(int, sys.stdin.readline().split())
    #     r = list(map(int, sys.stdin.readline().split()))
    #     A, B, C = [], [], []
    #     for _ in range(M):
    #         a, b, c = map(int, sys.stdin.readline().split())
    #         A.append(a)
    #         B.append(b)
    #         C.append(c)

    #     # Initialize distance matrix
    #     INF = float('inf')
    #     dist = [[INF for _ in range(N+1)] for _ in range(N+1)]
    #     for i in range(1, N+1):
    #         dist[i][i] = 0

    #     # Set initial distances
    #     for i in range(M):
    #         a, b, c = A[i], B[i], C[i]
    #         dist[a][b] = c
    #         dist[b][a] = c

    #     # Floyd-Warshall algorithm
    #     for k in range(1, N+1):
    #         for i in range(1, N+1):
    #             for j in range(1, N+1):
    #                 if dist[i][k] != INF and dist[k][j] != INF:
    #                     dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    #     # Generate all permutations of R towns
    #     min_dist = INF
    #     for perm in permutations(r):
    #         total = 0
    #         for i in range(R-1):
    #             total += dist[perm[i]][perm[i+1]]
    #         if total < min_dist:
    #             min_dist = total

    #     # Output the minimum distance
    #     print(min_dist)

    # if __name__ == "__main__":
    #     main()
    # """



    # public_tests= {"input": ["3\n4 5\n6 3\n10 2\n"], "output": ["5\n3 4\n4 4 1 2\n"]}
    # metadata = {
    #     "public_tests": public_tests,
    # }
    # reward = RewardCodeFn(RewardConfig)
    # input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata)
    # output = reward(input)
    # print(f"codetest output:{output}")

    #test app/taco
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
        print(5) 
    else: 
        print(5)
if __name__ == "__main__":
    main()
    """
    print(f"test the code_forces")
    #test_cases = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 20\n2 1 1", "output": "-1" }, { "input": "50 10000\n5 4 10 9 9 6 7 7 7 3 3 7 7 4 7 4 10 10 1 7 10 3 1 4 5 7 2 10 10 10 2 3 4 7 6 1 8 4 7 3 8 8 4 10 1 1 9 2 6 1", "output": "1943" }, { "input": "50 10000\n4 7 15 9 11 12 20 9 14 14 10 13 6 13 14 17 6 8 20 12 10 15 13 17 5 12 13 11 7 5 5 2 3 15 13 7 14 14 19 2 13 14 5 15 3 19 15 16 4 1", "output": "1891" }]
    test_cases = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 10\n3 2 1", "output": "5" } ] 
    metadata = {
         "test_cases": test_cases,
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata)
    output = reward(input)
    print(f"code_forces output:{output}")