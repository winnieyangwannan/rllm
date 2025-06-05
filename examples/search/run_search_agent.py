import asyncio

import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

from rllm.agents.tool_agent import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment


def load_search_r1_data(n=1, train_size=3000, test_size=100):
    if DatasetRegistry.dataset_exists("search_r1_combined", "test"):
        test_dataset = DatasetRegistry.load_dataset("search_r1_combined", "test")
        return test_dataset.get_data()
    
    print("Loading HotpotQA dataset...")
    
    hotpot_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)
    hotpot_train = hotpot_dataset["train"]
    hotpot_val = hotpot_dataset["validation"]
    
    print("Loading Natural Questions dataset...")
    
    nq_dataset = load_dataset("sentence-transformers/natural-questions", "pair")
    nq_train = nq_dataset["train"]
    
    hotpot_train_subset = hotpot_train.select(range(min(train_size // 2, len(hotpot_train))))
    hotpot_val_subset = hotpot_val.select(range(min(test_size // 2, len(hotpot_val))))
    nq_subset = nq_train.select(range(min(train_size // 2, len(nq_train))))
    
    def process_hotpot_example(example, idx, split):
        question = example["question"]
        ground_truth = example["answer"]
        data_source = "hotpotqa"
        
        task = {
            "question": question,
            "ground_truth": ground_truth,
            "data_source": data_source
        }
        
        return {
            "data_source": data_source,
            "prompt": [{
                "role": "user", 
                "content": f"Please answer the following question by searching for relevant information: {question}"
            }],
            "ability": "multi-hop-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "task": task,
                "tools": ["google_search"],
                "uid": f"hotpot_{example.get('id', idx)}",
                "question_type": example.get("type", "bridge"),
                "level": example.get("level", "medium")
            },
            "task": task,
            "uid": f"hotpot_{example.get('id', idx)}"
        }
    
    def process_nq_example(example, idx, split):
        question = example["query"]
        ground_truth = example["answer"][:200] + "..." if len(example["answer"]) > 200 else example["answer"]
        data_source = "natural_questions"
        
        task = {
            "question": question,
            "ground_truth": ground_truth,
            "data_source": data_source
        }
        
        return {
            "data_source": data_source,
            "prompt": [{
                "role": "user", 
                "content": f"Please answer the following question by searching for relevant information: {question}"
            }],
            "ability": "fact-retrieval",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "task": task,
                "tools": ["google_search"],
                "uid": f"nq_{idx}",
                "question_type": "factual",
                "level": "easy"
            },
            "task": task,
            "uid": f"nq_{idx}"
        }
    
    print("Processing HotpotQA training data...")
    hotpot_train_processed = [process_hotpot_example(example, idx, "train") for idx, example in enumerate(hotpot_train_subset)]
    
    print("Processing HotpotQA validation data...")
    hotpot_val_processed = [process_hotpot_example(example, idx, "test") for idx, example in enumerate(hotpot_val_subset)]
    
    print("Processing Natural Questions data...")
    nq_processed = [process_nq_example(example, idx, "train") for idx, example in enumerate(nq_subset)]
    
    # Combine datasets
    train_processed = hotpot_train_processed + nq_processed
    
    remaining_nq_size = min(test_size - len(hotpot_val_processed), len(nq_train) - len(nq_subset))
    if remaining_nq_size > 0:
        nq_test_subset = nq_train.select(range(len(nq_subset), len(nq_subset) + remaining_nq_size))
        nq_test_processed = [process_nq_example(example, idx + len(nq_subset), "test") for idx, example in enumerate(nq_test_subset)]
        test_processed = hotpot_val_processed + nq_test_processed
    else:
        test_processed = hotpot_val_processed
    
    print(f"Combined dataset: {len(train_processed)} train examples, {len(test_processed)} test examples")
    
    DatasetRegistry.register_dataset("search_r1_combined", train_processed, "train")
    test_dataset = DatasetRegistry.register_dataset("search_r1_combined", test_processed, "test")
    
    return test_dataset.get_data()

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 64

    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    envs = [
        ToolEnvironment(tools=["google_search"]) for _ in range(n_parallel_agents)
    ]

    agents = [
        ToolAgent(tools=envs[i].tools.tools, model_name=model_name, parser_name='qwen') 
        for i in range(n_parallel_agents)
    ]

    sampling_params = {
        "temperature": 0.6, 
        "top_p": 0.95, 
        "model": model_name
    }

    engine = AsyncAgentExecutionEngine(
        agents=agents,
        envs=envs,
        rollout_engine=None,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=16384,
        max_prompt_length=4096,
        config=None,
        n_parallel_agents=n_parallel_agents,
        enable_thinking=True,
    )

    tasks = load_search_r1_data(n=1)

    env_tasks = [item["task"] for item in tasks]

    results = asyncio.run(engine.execute_tasks(env_tasks)) 
