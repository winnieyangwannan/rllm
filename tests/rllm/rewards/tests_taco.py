import json 
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from rllm.rewards import RewardConfig, RewardInput, RewardType
from rllm.rewards.code_reward import RewardCodeFn

def _process_case__taco(i, data):
    model_response = f"""```python\n{data["solutions"]}\n```"""
    tests = data["tests"]
    problem = data["problem"]
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(
        problem="", 
        problem_type=RewardType.CODE, 
        model_response=model_response, 
        metadata=tests, 
        data_source="taco"
    )
    output = reward(input)
    failed = None 
    if not output.is_correct :
        failed = {
            "model_response": model_response,
            "tests": tests,
            "problem": problem,
        }
    return i, output, failed


def test_reward_taco(data):
    results = {}
    failed_cases = []
    failure_log_path = os.path.expanduser("~/rllm/rllm/data/train/code/failed_taco_tests.json")
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_process_case__taco, i, data[i]): i for i in range(len(data))}
        for future in as_completed(futures):
            try:
                id, output, failed= future.result()
                results[id] = output
                if failed is not None:
                    failed_cases.append(failed)
            except Exception as e:
                print(f"Error processing item: {e}")

    #save the  failed case to a JSON file if any 
    if failed_cases:
        with open(failure_log_path, "w") as f:
            json.dump(failed_cases, f)

    # Return the output corresponding to the last processed index
    return results[len(data) - 1]

if __name__ == "__main__":
    path =  os.path.expanduser("~/rllm/rllm/data/train/code/taco.json")
    with open(path, "r") as f:
        data = json.load(f)
    test_reward_taco(data)
