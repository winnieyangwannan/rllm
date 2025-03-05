import json 
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from rllm.rewards import RewardConfig, RewardInput, RewardType
from rllm.rewards.code_reward import RewardCodeFn
from rllm.data.utils import load_dataset
from rllm.data.dataset_types import TrainDataset


def _process_case_leetcode(i, entry):
    model_response = f"""
```python
{entry["completion"]}
```
"""
    tests = entry["tests"]
    reward = RewardCodeFn(RewardConfig)
    input_obj = RewardInput(
        problem="",
        problem_type=RewardType.CODE,
        model_response=model_response,
        metadata=tests,
        data_source="leetcode"
    )
    output = reward(input_obj)
    failed = None
    if not output.is_correct:
        failed = {
            "index": i,
            "problem": entry["problem"],
            "model_response": model_response,
            "tests": tests,
        }
    return i, output, failed


def _process_case_taco(i, data):
    """
    Process a single test case from the TACO dataset.
    
    Args:
        i: Index of the test case
        data: Test case data containing solutions and tests
        
    Returns:
        tuple: (index, reward output, failed case data if applicable)
    """
    model_response = f"""```python\n{data["solutions"]}\n```"""
    tests = data["tests"]
    reward = RewardCodeFn(RewardConfig)
    input_obj = RewardInput(
        problem="", 
        problem_type=RewardType.CODE, 
        model_response=model_response, 
        metadata=tests, 
        data_source="taco"
    )
    output = reward(input_obj)
    failed = None 
    if not output.is_correct:
        failed = {
            "problem": data["problem"],
            "model_response": model_response,
            "tests": tests,
        }
    return i, output, failed


def test_batched_reward(dataset: str):
    """
    Test the reward function on the TACO dataset.
    
    Processes all test cases in parallel and logs any failures.
    
    Returns:
        The reward output for the last test case.
    """
    if dataset == "taco":
        data = load_dataset(TrainDataset.Code.TACO)
        test_fn = _process_case_taco
    elif dataset == "leetcode":
        data = load_dataset(TrainDataset.Code.LIVECODEBENCH)
        test_fn = _process_case_leetcode
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    results = {}
    failed_cases = []
    failure_log_path = os.path.join(os.path.dirname(__file__), f"./{dataset}_test_err.json")
    counter = 0
    debug = False
    with ThreadPoolExecutor(max_workers=256) as executor:
        futures = [executor.submit(test_fn, i, data[i]) for i in range(len(data))]
        for future in as_completed(futures):
            try:
                idx, output, failed = future.result()
                results[idx] = output
                if failed is not None:
                    failed_cases.append(failed)
            except Exception as e:
                print(f"Error processing item: {e}")
            counter += 1
            if debug:
                print(counter)

    # Save the failed cases to a JSON file if any 
    if failed_cases:
        with open(failure_log_path, "w") as f:
            json.dump(failed_cases, f, indent=4)

    # Return the output corresponding to the last processed index
    return results[len(data) - 1]


if __name__ == "__main__":
    test_batched_reward(dataset="taco")
