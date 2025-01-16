import random

from vllm import SamplingParams

from rllm.data.load_dataset import Datasets, load_dataset

from rllm.rollout.distributed import DistributedVLLM
from rllm.rollout.distributed_client import DistributedLLMClient
from rllm.rollout.model import COTRollout
from rllm.system_prompts import COT_MATH_SYSTEM_PROMPT

from rllm.grading.math.sympy_checker import grade_answer
from rllm.trainer.reinforce import ReinforceTrainer, ReinforceConfig

import asyncio

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

async def async_generate_responses(model_name, queries):
    messages_batch = [
        [
            {"role": "system", "content": COT_MATH_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        for query in queries
    ]

    endpoints = [
        {"url": "http://0.0.0.0:8000/v1/chat/completions", "weight": 1.0},
    ]

    sampling_params = {
        "temperature": 0.7,
        "model": model_name,
        "max_tokens": 100
    }

    try:
        client = await DistributedLLMClient.create_client(endpoints)
        results = await client.chat(messages_batch, sampling_params)
        return [result["choices"][0]["message"]["content"] for result in results]
    except Exception as e:
        print(f"An error occurred during response generation: {e}")
        return []
    finally:
        if "client" in locals():
            await client.__aexit__(None, None, None)


def generate_responses(model_name, queries):
    return asyncio.run(async_generate_responses(model_name, queries))


# def generate_responses(model_name, queries):
#     engine = DistributedVLLM(num_workers=1, tensor_parallel_size=4, model=model_name)
#     model = COTRollout(engine=engine)

#     responses = model.rollout(queries)
#     return responses


def compute_rewards(responses, labels):
    rewards = []
    for response, label in zip(responses, labels):
        is_correct = grade_answer(response, label)
        reward = 1.0 if is_correct else -1.0
        rewards.append(reward)
    return rewards


def train():
    batch_size = 4
    n_rollouts = 1

    model_name = "Qwen/QwQ-32B-Preview"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    trainer = ReinforceTrainer(model, tokenizer, ReinforceConfig())
    dataset = load_dataset(Datasets.OMNI)

    for i in range(2):
        batch = dataset[i * batch_size : (i + 1) * batch_size]

        queries = [entry["problem"] for entry in batch]
        labels = [entry["answer"] for entry in batch]
        responses = generate_responses(model_name, queries)
        rewards = compute_rewards(responses, labels)

        print(responses)

        stats = trainer.step({"queries": queries, "responses": responses, "rewards": rewards})

        print(stats)
        trainer.save(f"./ckpts/qwq_iter{i}")
        model_name = f"./ckpts/qwq_iter{i}"


if __name__ == "__main__":
    train()