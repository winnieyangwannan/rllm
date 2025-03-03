import json 
from concurrent.futures import ThreadPoolExecutor, as_completed

from rllm.rewards import RewardConfig, RewardInput, RewardType
from rllm.rewards.code_reward import RewardCodeFn

def test_reward_taco(data):
    model_response = data["solutions"]
    metadata = data["tests"]
    id = data["id"]
    problem = data["problem"]
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="taco")
    output = reward(input)
    if not output.is_correct :
        print(f"test_reward_taco Failed test case: {id} and model_response:\n{model_response}\ntests:{metadata}")
    return output.is_correct, id


def process_data_with_threads(data, num_threads=256):
    results = []
    false_result = []
    false_id = [] 
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(test_reward_taco, item) for item in data]
        for future in as_completed(futures):
            try:
                result= future.result()
                output = result[0]
                id = result[1]
                if output == False:
                    false_result.append(result)
                    false_id.append(id)
                results.append(result)
            except Exception as e:
                print(f"Error processing item: {e}")
    print(f"Total test cases: {len(data)} and False test cases: {len(false_result)}")
    #save the false_id 
    with open("../../../rllm/data/train/code/false_id_taco.json", "w") as f:
        json.dump(false_id, f)
    return results

if __name__ == "__main__":
    path = "../../../rllm/data/train/code/taco.json"
    with open(path, "r") as f:
        data = json.load(f)
    data = data
    results = process_data_with_threads(data)
    print(f"Total test cases: {len(data)}")
