from rllm.system_prompts import COT_SYSTEM_PROMPT, COT_MATH_SYSTEM_PROMPT
from rllm.data.load_dataset import Datasets, load_dataset
# If grade_answer is not used in this snippet, you can remove the import.
from rllm.grading.grader import grade_answer
from rllm.rollout.distributed import DistributedVLLM

from vllm import SamplingParams

import requests
import time
import json

def poll_vllm_chat_completions(api_url, payload):
    """
    Polls the vllm chat completions API to fetch the output.

    Parameters:
    - api_url (str): The API endpoint for chat completions.
    - payload (dict): The payload to send to the API, including the prompt or input message.
    - interval (int): Polling interval in seconds (default: 1 second).
    - timeout (int): Maximum time to wait for a response (default: 30 seconds).

    Returns:
    - dict: The response from the API, or None if timeout is reached.
    """
    try:
        # Send the initial request to start processing
        response = requests.post(f"{api_url}/v1/chat/completions", json=payload)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return None

        response_data = response.json()
        choices = response_data.get("choices", [])
        if not choices:
            print("Error: No choices found in the response.")
            return None

        messages = choices[0]
        if not messages:
            print("Error: No messages found in the response.")
            return None
        return messages['message']['content']

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    #aime_problems = load_dataset(Datasets.AIME)
    

    # Load from aime in data/aime_v1.json
    api_url = "http://0.0.0.0:8000"  # Replace with your vllm server URL
    content_dict = [
        {"role": "system", "content": COT_MATH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Find all real $ a$, such that there exist a function $ f: \\mathbb{R}\\rightarrow\\mathbb{R}$ satisfying the following inequality:\n\\[ x\\plus{}af(y)\\leq y\\plus{}f(f(x))\n\\]\nfor all $ x,y\\in\\mathbb{R}$"
            },
    ]
    payload = {
        "messages": content_dict,
        "model": "Qwen/QwQ-32B-Preview",  # Replace with your model name if different
    }

    # engine = DistributedVLLM(num_workers=2, tensor_parallel_size=2, model="Qwen/QwQ-32B-Preview")
    # responses = engine.chat([payload["messages"]], SamplingParams(temperature=1.0))
    # print(responses)
    # engine.shutdown(persist=True)

    response = poll_vllm_chat_completions(api_url, payload)

    content_dict.append({"role": "assistant", "content": response})
    import pprint
    pprint.pprint(content_dict)
    print(grade_answer(response, "f(x) = 2x"))
