import json 
from concurrent.futures import ThreadPoolExecutor, as_completed

from rllm.rewards import RewardConfig, RewardInput, RewardType
from rllm.rewards.code_reward import RewardCodeFn

def test_reward_taco(data):
    print(f"type(data): {type(data)}")
    model_response = data["solutions"]
    metadata = data["tests"]
    id = data["id"]
    print(f"model_response:\n{model_response}")
    print(f"metadata:\n{metadata}")
    problem = data["problem"]
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="taco")
    output = reward(input)
    if not output.is_correct :
        print(f"test_reward_taco Failed test case: {id} and model_re")
        # print(f"test_reward_taco Failed test case solutions:\n{model_response}")
        # print(f"test_reward_taco Failed test case input_output:\n{metadata}")
    # assert output.is_correct == True, "Reward is not correct"
    return output.is_correct, id


def process_data_with_threads(data, num_threads=256):
    results = []
    false_result = []
    false_id = [] 
    for item in data:
        result = test_reward_taco(item)
        output = result[0]
        id = result[1]
        if output == False:
            false_result.append(result)
            print(f"false_id: {id}")
            false_id.append(id)
        results.append(result)
    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     futures = [executor.submit(test_reward_taco, item) for item in data]
    #     for future in as_completed(futures):
    #         try:
    #             result= future.result()
    #             output = result[0]
    #             id = result[1]
    #             if output == False:
    #                 false_result.append(result)
    #                 false_id.append(id)
    #             results.append(result)
    #         except Exception as e:
    #             print(f"Error processing item: {e}")

    print(f"Total test cases: {len(data)} and False test cases: {len(false_result)}")
    return results

if __name__ == "__main__":
    path = "false_taco.json"
    with open(path, "r") as f:
        data = json.load(f)
    # data = data[:3][2:]
    data = data[:10]
    print(f"len(data): {len(data)} and type(data):{type(data)}")
    print(f"data:{data}")
    results = process_data_with_threads(data)
    print(f"Total test cases: {len(data)}")
