import asyncio
import json
import random
from copy import deepcopy

from openai import AsyncOpenAI

from rllm.tools import PythonInterpreter, ToolCaller
from rllm.tools.utils import chat_completion_with_tool


def load_data(n=8):
    dataset = load_dataset(TrainDataset.Math.AIME)
    # print(len(dataset))
    # random.seed(42)
    # random.shuffle(dataset)
    # dataset = dataset[:2000]  # Randomly sample 2000 examples
    data = []
    # First collect all examples
    for idx, example in enumerate(dataset):
        processed = process_fn(example, idx)
        for i in range(n):
            data.append(deepcopy(processed))
    return data


def process_fn(example, idx):
    question = example.pop("problem")
    instruction = "Let's think step by step, put your final answer within \\boxed{}, and write python to evaluate math expressions if needed."
    question = f"{question} {instruction}"
    answer = example.pop("answer")

    data = {
        "messages": [
            {"role": "user", "content": question},
        ],
        "answer": answer,
        "problem": question,
        "idx": idx
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

    data = load_data(n=8)

    # This is for model served with vLLM.
    client = AsyncOpenAI(
        base_url="http://localhost:30000/v1",
        api_key="EMPTY",
    )

    interpreter = PythonInterpreter(n_sandboxes=16)
    tool_caller = ToolCaller(tools=[interpreter], parser_type="python")

    results = chat_completion_with_tool(
        client, tool_caller, data, model="/data/sijun/checkpoints/deepscaler-toolcall-claude-python/", batch_size=64
    )

    # evaluate_results(results)
    # print("answer:", dataset[idx]['answer'])
