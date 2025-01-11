from rllm.system_prompts import COT_SYSTEM_PROMPT, COT_MATH_SYSTEM_PROMPT
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
    # Load from aime in data/aime_v1.json
    api_url = "http://0.0.0.0:8000"  # Replace with your vllm server URL
    
    payload = {
        "messages": [
            {"role": "system", "content": COT_MATH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Find all real $ a$, such that there exist a function $ f: \\mathbb{R}\\rightarrow\\mathbb{R}$ satisfying the following inequality:\n\\[ x\\plus{}af(y)\\leq y\\plus{}f(f(x))\n\\]\nfor all $ x,y\\in\\mathbb{R}$", "solution": "\nWe are tasked with finding all real values of \\( a \\) such that there exists a function \\( f: \\mathbb{R} \\rightarrow \\mathbb{R} \\) satisfying the inequality:\n\\[\nx + af(y) \\leq y + f(f(x))\n\\]\nfor all \\( x, y \\in \\mathbb{R} \\).\n\n### Step-by-step Analysis\n\n1. **Case Analysis:**\n\n   Consider the inequality for specific choices of \\( y \\):\n\n   - **Choice:** \\( y = x \\)\n   \n     Substitute \\( y = x \\) in the inequality:\n     \\[\n     x + af(x) \\leq x + f(f(x))\n     \\]\n     Simplifying, we have:\n     \\[\n     af(x) \\leq f(f(x))\n     \\]\n     This must hold for all \\( x \\in \\mathbb{R} \\).\n\n   - **Exploration for Special Values of \\( a \\):**\n\n     Suppose \\( a = 1 \\). Then the inequality becomes:\n     \\[\n     x + f(y) \\leq y + f(f(x))\n     \\]\n\n     If \\( f(x) = x \\) is a solution for all \\( x \\), the inequality simplifies to:\n     \\[\n     x + y \\leq y + x\n     \\]\n     which is trivially true. Therefore, \\( a = 1 \\) is a valid solution.\n\n2. **Consider Other Restrictions:**\n\n   - Consider the contrapositive cases where the inequality might fail for choices of \\( a \\geq 2 \\):\n\n     If we assume \\( a \\geq 2 \\), the inequality:\n     \\[\n     af(x) - f(f(x)) \\leq 0\n     \\]\n     suggests \\( f(f(x)) \\) must generally dominate \\( af(x) \\), which might be restrictive. Testing:\n     \n     For \\( f(x) = x \\), the inequality would require \\( x \\leq 0 \\), restricting \\( x \\).\n   \n   Therefore, solutions for \\( a \\geq 2 \\) are non-trivial without further modification of \\( f(x) \\).\n\n3. **Conclude with Restrictions on \\( a \\):**\n\n   - Given potential limitations for \\( a \\geq 2 \\), explore possible other solutions where \\( a < 0 \\):\n   \n     For negative \\( a \\), say \\( a = -1 \\), the inequality becomes:\n     \\[\n     x - f(y) \\leq y + f(f(x))\n     \\]\n     This relation allows greater flexibility and potential for constructing \\( f(x) \\) that holds generally, as this rearranges to \\( f(y) \\geq y + x - f(f(x)) \\).\n\nIn conclusion, analyzing the cases:\n\n- \\( a = 1 \\) where function simplification \\( f(x) = x \\) holds trivially.\n- \\( a < 0 \\) allowing flexibility for function structures.\n  \nThese scenarios provide viable solutions. Thus, the values of \\( a \\) are:\n\\[\n\\boxed{a < 0 \\text{ or } a = 1}\n\\]\n",
            },
        ],
        "model": "Qwen/QwQ-32B-Preview",  # Replace with your model name if different
    }


    engine = DistributedVLLM(num_workers=2, tensor_parallel_size=2, model=payload["model"])

    responses = engine.chat(payload["messages"], SamplingParams(temperature=1.0))

    print(responses)

    # response = poll_vllm_chat_completions(api_url, payload)
    # print(response)
    # grade_answer(response, "f(x) = 2x")
