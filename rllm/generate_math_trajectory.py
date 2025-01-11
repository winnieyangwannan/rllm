from rllm.system_prompts import COT_SYSTEM_PROMPT, COT_MATH_SYSTEM_PROMPT
# If grade_answer is not used in this snippet, you can remove the import.
from rllm.grading.grader import grade_answer

import requests
import time
import json

def poll_vllm_chat_completions(api_url, payload, interval=1, timeout=30):
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

# Example usage
if __name__ == "__main__":
    # Load from aime in data/aime_v1.json
    api_url = "http://0.0.0.0:8000"  # Replace with your vllm server URL
    payload = {
        "messages": [
            {"role": "system", "content": COT_MATH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Let $x$, $y$, and $z$ all exceed $1$ and let $w$ be a positive number "
                    "such that $\\log_x w = 24$, $\\log_y w = 40$ and $\\log_{xyz} w = 12$. "
                    "Find $\\log_z w$."
                ),
            },
        ],
        "model": "Qwen/QwQ-32B-Preview",  # Replace with your model name if different
    }

    response = poll_vllm_chat_completions(api_url, payload)
    print(response)
    grade_answer(response, "60")
