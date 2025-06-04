import asyncio

from datasets import load_dataset
from transformers import AutoTokenizer

from rllm.agents.tool_agent import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment


def load_search_r1_data(n=1, train_size=3000, test_size=100):
    # Check if dataset already exists in registry
    if DatasetRegistry.dataset_exists("search_r1_combined", "test"):
        test_dataset = DatasetRegistry.load_dataset("search_r1_combined", "test")
        return test_dataset.get_data()
    
    # If not, load and combine HotpotQA and Natural Questions datasets
    print("Loading HotpotQA dataset...")
    
    # Load HotpotQA dataset (using distractor subset)
    hotpot_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)
    hotpot_train = hotpot_dataset["train"]
    hotpot_val = hotpot_dataset["validation"]
    
    print("Loading Natural Questions dataset...")
    
    # Load Natural Questions dataset 
    nq_dataset = load_dataset("sentence-transformers/natural-questions", "pair")
    nq_train = nq_dataset["train"]
    
    # Take subsets
    hotpot_train_subset = hotpot_train.select(range(min(train_size // 2, len(hotpot_train))))
    hotpot_val_subset = hotpot_val.select(range(min(test_size // 2, len(hotpot_val))))
    nq_subset = nq_train.select(range(min(train_size // 2, len(nq_train))))
    
    def process_hotpot_example(example, idx):
        """Convert HotpotQA example to unified format."""
        return {
            "question": example["question"],
            "ground_truth": example["answer"],
            "data_source": "hotpotqa",
            "index": idx,
            "uid": f"hotpot_{example.get('id', idx)}",
            "ability": "multi-hop-reasoning",
            "prompt": f"Question: {example['question']}",
            "question_type": example.get("type", "bridge"),
            "level": example.get("level", "medium")
        }
    
    def process_nq_example(example, idx):
        """Convert Natural Questions example to unified format."""
        return {
            "question": example["query"],
            "ground_truth": example["answer"][:200] + "..." if len(example["answer"]) > 200 else example["answer"],
            "data_source": "natural_questions", 
            "index": idx,
            "uid": f"nq_{idx}",
            "ability": "fact-retrieval",
            "prompt": f"Question: {example['query']}",
            "question_type": "factual",
            "level": "easy"
        }
    
    # Process all datasets
    print("Processing HotpotQA training data...")
    hotpot_train_processed = [process_hotpot_example(example, idx) for idx, example in enumerate(hotpot_train_subset)]
    
    print("Processing HotpotQA validation data...")
    hotpot_val_processed = [process_hotpot_example(example, idx) for idx, example in enumerate(hotpot_val_subset)]
    
    print("Processing Natural Questions data...")
    nq_processed = [process_nq_example(example, idx) for idx, example in enumerate(nq_subset)]
    
    # Combine datasets
    # For training: combine HotpotQA train + first half of NQ
    train_processed = hotpot_train_processed + nq_processed
    
    # For testing: use HotpotQA validation + second half of NQ (if we have enough)
    remaining_nq_size = min(test_size - len(hotpot_val_processed), len(nq_train) - len(nq_subset))
    if remaining_nq_size > 0:
        nq_test_subset = nq_train.select(range(len(nq_subset), len(nq_subset) + remaining_nq_size))
        nq_test_processed = [process_nq_example(example, idx + len(nq_subset)) for idx, example in enumerate(nq_test_subset)]
        test_processed = hotpot_val_processed + nq_test_processed
    else:
        test_processed = hotpot_val_processed
    
    print(f"Combined dataset: {len(train_processed)} train examples, {len(test_processed)} test examples")
    
    # Register the datasets with separate splits
    DatasetRegistry.register_dataset("search_r1_combined", train_processed, "train")
    test_dataset = DatasetRegistry.register_dataset("search_r1_combined", test_processed, "test")
    
    return test_dataset.get_data()

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 64

    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create environments first (following the pattern from run_tool_agent_math.py)
    # Use python tools instead of google_search to avoid HTTP connection issues
    envs = [
        ToolEnvironment(tools=["google_search"]) for _ in range(n_parallel_agents)
    ]

    # Create agents using the correct pattern
    agents = [
        ToolAgent(tools=envs[i].tools.tools, model_name=model_name, parser_name='qwen') 
        for i in range(n_parallel_agents)
    ]

    sampling_params = {
        "temperature": 0.6, 
        "top_p": 0.95, 
        "tools": envs[0].tools.json,  # Add tools to sampling params
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

    results = asyncio.run(engine.execute_tasks(tasks)) 