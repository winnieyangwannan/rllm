import asyncio
import json
from copy import deepcopy

from openai import AsyncOpenAI

from rllm.environments.tools import PythonInterpreter, ToolCaller
from rllm.environments.tools.utils import parse_tool_calls


async def _apply_tool(completion, messages, tool_caller, id=None):
    tool_calls = parse_tool_calls(completion.choices[0].message.content)

    if len(tool_calls) > 0:
        tool_call = tool_calls[0]
        if id is not None:
            tool_call["parameters"]["id"] = id
        tool_call_result = await tool_caller(tool_call["name"], tool_call["parameters"])
        print("tool_call_result", tool_call_result)
        messages.append(tool_call_result)
        return True

    return False


def chat_completion_with_tool(
    client: AsyncOpenAI,
    tool_caller: ToolCaller,
    messages_list,
    model="gpt-4",
    max_round=20,
    batch_size=32,  # Added batch_size parameter
):
    async def tool_call_flow(example, request_id):
        messages = example["prompt"]
        tool_infos = tool_caller.get_tool_infos()

        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_infos,
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
        )
        messages.append(
            {
                "role": "assistant",
                "content": completion.choices[0].message.content,
            }
        )
        print("round: 0", completion.choices[0].message.content)
        curr_round = 0
        while curr_round < max_round:
            use_tools = await _apply_tool(
                completion, messages, tool_caller, id=request_id
            )
            if use_tools:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_infos,
                    temperature=0.6,
                    max_tokens=4096,
                    top_p=0.95,
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": completion.choices[0].message.content,
                    }
                )
            else:
                break

            curr_round += 1
            print(f"round {curr_round}:", completion.choices[0].message.content)

        return example

    async def run_batch():
        # Initialize pool with first batch of requests
        active_requests = []
        results = []
        messages_iter = iter(messages_list)
        processed_count = 0
        request_id = 0

        # Fill initial pool
        for _ in range(batch_size):
            try:
                messages = next(messages_iter)
                task = asyncio.create_task(tool_call_flow(messages, request_id))
                active_requests.append((task, request_id))
                request_id += 1
            except StopIteration:
                break

        # Process requests and refill pool
        while active_requests:
            done, pending = await asyncio.wait(
                [task for task, _ in active_requests],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Update active_requests with pending tasks
            active_requests = [
                (task, id) for task, id in active_requests if task in pending
            ]

            for completed_task in done:
                # Find the ID for the completed task
                # task_id = next(id for task, id in active_requests if task == completed_task)
                final_messages = await completed_task
                # results.append({"id": task_id, "data": final_messages})
                results.append(final_messages)
                processed_count += 1

                # Save results checkpoint every 100 examples
                if processed_count % 100 == 0:
                    with open("messages_checkpoint.json", "w") as f:
                        json.dump(results, f, indent=2)

                # Try to add new request to maintain pool
                try:
                    messages = next(messages_iter)
                    new_task = asyncio.create_task(tool_call_flow(messages, request_id))
                    active_requests.append((new_task, request_id))
                    request_id += 1
                except StopIteration:
                    pass

            print("Active requests:", len(active_requests))

        return results

    return asyncio.run(run_batch())


def load_data(n=1):
    dataset = load_dataset(TestDataset.Math.AIME)
    data = []
    # First collect all examples
    for example in dataset:
        processed = process_fn(example)
        for i in range(n):
            data.append(deepcopy(processed))
    return data


def process_fn(example):
    question = example.pop("problem")
    instruction = "Let's think step by step, put your final answer within \\boxed{}, and use tools to evaluate math expressions if needed."
    question = f"{question} {instruction}"
    answer = example.pop("answer")

    data = {
        "prompt": [
            {"role": "user", "content": question},
        ],
        "answer": answer,
        "problem": question,
    }
    return data


def evaluate_results(results):
    from collections import defaultdict

    from rllm.rewards.math_utils.utils import extract_answer, grade_answer_verl

    # Create a map to store correct answers per problem
    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)

    # Count correct answers for each problem
    for entry in results:
        problem = entry["problem"]  # Use problem text as unique identifier
        # print("answer", entry['answer'])
        is_correct = grade_answer_verl(entry["prompt"][-1]["content"], entry["answer"])
        entry["is_correct"] = is_correct
        problem_correct_map[problem] += is_correct
        problem_total_map[problem] += 1

    # Calculate pass@1 and pass@16
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    pass_at_16 = (
        sum(1 for problem, correct in problem_correct_map.items() if correct > 0)
        / total_problems
    )

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print("Average Pass@16 Accuracy:", pass_at_16)


if __name__ == "__main__":
    from rllm.data.dataset_types import TestDataset, TrainDataset
    from rllm.data.utils import load_dataset

    data = load_data(n=1)

    # This is for model served with vLLM.
    client = AsyncOpenAI(
        base_url="http://0.0.0.0:8081/v1",
        api_key="EMPTY",
    )

    interpreter = PythonInterpreter(n_sandboxes=16)
    tool_caller = ToolCaller(tools=[interpreter])

    results = chat_completion_with_tool(
        client, tool_caller, data, model="deepscaler-toolcall", batch_size=16
    )

    evaluate_results(results)
    # print("answer:", dataset[idx]['answer'])
