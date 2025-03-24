import json

from rllm.data.dataset_types import TrainDataset
from rllm.data.utils import load_dataset
from rllm.globals import THOUGHT_DELIMITER_START
from rllm.rewards.rl_reward import rllm_reward_fn
from rllm.sampler import DistributedSampler
from rllm.sampler.sampler_types import SampleBatch


def process_fn(example):
    question = example.pop('problem')
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    question = f"{question} {instruction}"
    answer = example.pop('answer')

    data = {
        "prompt": [{
            "role": "user",
            "content": question
        }],
        "answer": answer,
        "problem": question,
    }
    return data


def convert_to_sharegpt_format(example, sample_batch: SampleBatch):
    sharegpt_data = []
    for sample in sample_batch.samples:
        if sample.is_correct:
            conversation = []
            
            conversation.append({
                "from": "human",
                "value": example['problem']
            })
            
            # append <think> token at the start if it does not exist.
            if THOUGHT_DELIMITER_START not in sample.response:
                sample.response = THOUGHT_DELIMITER_START + sample.response

            conversation.append({
                "from": "gpt",
                "value": sample.response
            })

            sharegpt_data.append({
                "conversations": conversation,
                "tools": "[]"  # Empty tools list since we're not using tools in this case
            })
    
    return sharegpt_data


async def generate_batch_trajectories(batch, sampler):
    response_futures = []
    for example in batch:
        response_futures.append(sampler.chat_completion(example['prompt']))
    
    # Gather all responses
    responses = await asyncio.gather(*response_futures)

    trajectories = convert_to_sharegpt_format(responses)
            
    return trajectories


async def generate_trajectories(data_iterator, sampler, batch_size=16, **sampler_kwargs):
    # Initialize the pool with first batch of requests
    active_requests = []
    results = []
    
    async def create_request(example):
        sample_batch = await sampler.chat_completion(example['prompt'], **sampler_kwargs)
        for sample in sample_batch.samples:
            sample.is_correct = rllm_reward_fn(sample.response, ground_truth=example['answer'])
        return example, sample_batch

    # Initialize the pool
    for _ in range(batch_size):
        try:
            example = next(data_iterator)
            task = asyncio.create_task(create_request(example))
            active_requests.append(task)
        except StopIteration:
            break

    while active_requests:
        done, pending = await asyncio.wait(active_requests, return_when=asyncio.FIRST_COMPLETED)
        active_requests = list(pending)
        
        for task in done:
            example, sample_batch = await task
            results.extend(convert_to_sharegpt_format(example, sample_batch))
            
            # Try to add a new request to maintain the pool
            try:
                example = next(data_iterator)
                new_task = asyncio.create_task(create_request(example))
                active_requests.append(new_task)
            except StopIteration:
                pass

        print("num active requests:", len(active_requests))
    
    print("final result:", len(results))
    return results


def load_data():
    dataset = load_dataset(TrainDataset.DEEPSCALER)
    train_data = []
    for example in dataset:
        train_data.append(process_fn(example))
    return train_data
    

async def main(args):
    data = load_data()[:4]

    sampler = DistributedSampler(
        num_workers=args.num_workers,
        tensor_parallel_size=args.tensor_parallel_size,
        backend=args.backend,
        model=args.model
    )

    trajectories = await generate_trajectories(iter(data), sampler, batch_size=args.batch_size, temperature=args.temperature, n=args.n)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for generating trajectories")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of workers for distributed sampling")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size for model parallelism")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "sglang"],
                        help="Backend for sampling (vllm or sglang)")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Model to use for sampling")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of completions to generate per prompt")

    args = parser.parse_args()
    asyncio.run(main(args))