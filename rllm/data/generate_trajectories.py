from rllm.system_prompts import COT_SYSTEM_PROMPT, COT_MATH_SYSTEM_PROMPT
from rllm.data.load_dataset import Datasets, load_dataset
from rllm.rewards.math.sympy_checker import grade_answer
from rllm.rollout.distributed import DistributedVLLM

from vllm import SamplingParams

import requests
import json
import pprint
import os

def poll_vllm_chat_completions(api_url, payload):
    """
    Polls the vllm chat completions API to fetch the output.

    Parameters:
    - api_url (str): The API endpoint for chat completions.
    - payload (dict): The payload to send to the API, including the prompt or input message.

    Returns:
    - str or None: The assistant's response (assistant content).
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
    # Very inefficient, will just run for tonight to test things out.
    # Load from AIME in data/aime_v1.json
    api_url = "http://0.0.0.0:8000"  # Replace with your vllm server URL
    aime_problems = load_dataset(Datasets.AIME)

    # Path to the JSON file where we'll store cumulative data
    output_file = "aime_responses.json"

    # 1. Load existing JSON data if file already exists
    output_data = []

    # 2. Process each problem and incrementally save to JSON
    for aime in aime_problems:
        problem = aime['problem']

        content_dict = [
            {"role": "system", "content": COT_MATH_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]

        payload = {
            "messages": content_dict,
            "model": "Qwen/QwQ-32B-Preview",  # Replace with your model name if different
        }

        # Example: If using DistributedVLLM, you'd do something like:
        # engine = DistributedVLLM(num_workers=2, tensor_parallel_size=2, model="Qwen/QwQ-32B-Preview")
        # responses = engine.chat([payload["messages"]], SamplingParams(temperature=1.0))
        # engine.shutdown(persist=True)

        # Poll the server
        response = poll_vllm_chat_completions(api_url, payload)

        # Add assistant response to the conversation
        if response is not None:
            content_dict.append({"role": "assistant", "content": response})

        # Print conversation for debugging
        pprint.pprint(content_dict)

        # Grade the answer
        grader_result = grade_answer(response if response else "", str(aime['answer']))
        print("Grader Result:", grader_result)

        # Convert grader result to 1 or 0
        # Adjust logic based on how your grader returns results
        grade_value = 1 if grader_result else 0

        # Make a copy of the original AIME record
        record = dict(aime)  # so we preserve the original fields
        record["trajectory"] = content_dict
        record["grade"] = grade_value

        # Add to our cumulative data
        output_data.append(record)

        # 3. Save the updated data back to the file **after each iteration**
        with open(output_file, mode="w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("All problems processed and appended to JSON.")
