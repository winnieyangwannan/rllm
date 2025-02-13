from rllm.rewards import RewardConfig, RewardInput, RewardType
from rllm.rewards.code_reward import RewardCodeFn

# #test the code_forces
if __name__ == "__main__":
    model_response = """
    import sys
    from itertools import permutations
    def main():
        n,m=map(int, input().split()) 
        a=sum(list(map(int, input().split()))) 
        if a+(n-1)*10<=m: 
            print((m-a)//5) 
        else: 
            print(-1)
    if __name__ == "__main__":
        main()
    """
    test_cases = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 20\n2 1 1", "output": "-1" }, { "input": "50 10000\n5 4 10 9 9 6 7 7 7 3 3 7 7 4 7 4 10 10 1 7 10 3 1 4 5 7 2 10 10 10 2 3 4 7 6 1 8 4 7 3 8 8 4 10 1 1 9 2 6 1", "output": "1943" }, { "input": "50 10000\n4 7 15 9 11 12 20 9 14 14 10 13 6 13 14 17 6 8 20 12 10 15 13 17 5 12 13 11 7 5 5 2 3 15 13 7 14 14 19 2 13 14 5 15 3 19 15 16 4 1", "output": "1891" }]

    metadata = {
        "test_cases": test_cases,
        "dataset_flag": "codeforces",
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata)
    output = reward(input)