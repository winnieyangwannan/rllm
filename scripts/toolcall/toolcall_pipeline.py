import asyncio
import json

from openai import AsyncOpenAI

from rllm.data.dataset_types import TestDataset
from rllm.data.utils import load_dataset
from rllm.environments.tools import PythonInterpreter
from rllm.environments.tools.utils import parse_tool_calls
from rllm.sampler import DistributedSampler
from rllm.environments.tools import ToolCaller

async def _apply_tool(completion, messages, tool_caller):
    tool_call = parse_tool_calls(completion.samples[0].response)

    if tool_call:
        tool_call_result = await tool_caller(
            tool_call["name"], tool_call["parameters"]
        )
        tool_call_result['json_str'] = tool_call['json_str']
        messages.append(tool_call_result)
        return True
    return False


def chat_completion_with_tool(
    sampler: DistributedSampler,
    tool_caller: ToolCaller,
    dataset,
    max_round=20,
    **sampler_kwargs,
):
    async def tool_call_flow(example):
        messages = example['prompt']

        tool_infos = tool_caller.get_tool_infos()

        completion = await sampler.chat_completion(
            messages, tools=tool_infos, **sampler_kwargs
        )
        messages.append(
            {
                "role": "assistant",
                "content": completion.samples[0].response,
            }
        )
        print("round: 0", completion.samples[0].response)
        curr_round = 0
        while curr_round < max_round:
            use_tools = await _apply_tool(completion, messages, tool_caller)
            if use_tools:
                completion = await sampler.chat_completion(
                    messages=messages, tools=tool_infos, **sampler_kwargs
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": completion.samples[0].response,
                    }
                )
            else:
                break

            curr_round += 1
            print(f"round {curr_round}:", completion.samples[0].response)

        return example

    async def run_batch():
        # Initialize pool with first batch of requests
        active_requests = []
        results = []
        dataset_iter = iter(dataset)
        processed_count = 0

        # Fill initial pool
        for _ in range(args.batch_size):
            try:
                example = next(dataset_iter)
                task = asyncio.create_task(tool_call_flow(example))
                active_requests.append(task)
            except StopIteration:
                break

        # Process requests and refill pool
        while active_requests:
            done, pending = await asyncio.wait(
                active_requests, return_when=asyncio.FIRST_COMPLETED
            )
            active_requests = list(pending)

            for task in done:
                final_example = await task
                results.append(final_example)
                processed_count += 1

                # Save results every 100 examples
                if processed_count % 100 == 0:
                    with open("results_checkpoint.json", "w") as f:
                        json.dump(results, f, indent=2)

                # Try to add new request to maintain pool
                try:
                    example = next(dataset_iter)
                    new_task = asyncio.create_task(tool_call_flow(example))
                    active_requests.append(new_task)
                except StopIteration:
                    pass

            print("Active requests:", len(active_requests))

        return results

    return asyncio.run(run_batch())


def load_data():
    dataset = load_dataset(TestDataset.Math.AIME)
    data = []
    for example in dataset:
        data.append(process_fn(example))
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
    from rllm.rewards.math_utils.utils import grade_answer_verl

    n_correct = 0
    for entry in results:
        is_correct = grade_answer_verl(entry['prompt'][-1]['content'], entry['answer'])
        entry['is_correct'] = is_correct
        n_correct += is_correct

    print("Total problems:", len(results))
    print("Average Pass@1 Accuracy:", n_correct/len(results))


def main(args):
    data = load_data()

    sampler = DistributedSampler(
        num_workers=args.num_workers,
        tensor_parallel_size=args.tensor_parallel_size,
        backend=args.backend,
        model=args.model,
        chat_template=args.chat_template,
        enable_auto_tool_choice=True,
        tool_call_parser=args.tool_call_parser,
        tokenizer=args.tokenizer
    )

    interpreter = PythonInterpreter()
    tool_caller = ToolCaller(tools=[interpreter])

    results = chat_completion_with_tool(sampler, tool_caller, data, args.temperature)

    evaluate_results(results)

    with open("results.json", "w+") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for generating trajectories",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers for distributed sampling",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for model parallelism",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "sglang"],
        help="Backend for sampling (vllm or sglang)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="agentica-org/DeepScaleR-1.5B-Preview",
        help="Model to use for sampling",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="agentica-org/DeepScaleR-1.5B-Preview",
        help="Tokenizer to use for sampling"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument(
        "--chat-template", type=str, default="../../rllm/templates/r1-toolcall-qwen.jinja", help="Chat templates for tool call"
    )
    parser.add_argument(
        "--tool-call-parser", type=str, default="hermes", help="Tool call parser for vLLM"
    )

    args = parser.parse_args()
    main(args)
