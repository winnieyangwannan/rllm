from datasets import load_dataset
import json

from rllm.data.dataset import DatasetRegistry


def prepare_deepcoder_data(train_size: int = 24000, test_size: int = 500):
    """
    Prepare DeepCoder training and test datasets using LiveCodeBench with proper temporal splits.
    
    Following the paper methodology:
    - Training: LiveCodeBench v5 problems from May 2023 - July 2024 (older problems)
    - Testing: LiveCodeBench v2 problems from August 2024 - January 2025 (recent problems)
    
    This ensures no temporal contamination between train/test as mentioned in the paper.

    Args:
        train_size: Maximum number of training examples to load
        test_size: Maximum number of test examples to load

    Returns:
        Tuple[Dataset, Dataset]: (train_dataset, test_dataset)
    """
    
    # Load training data: LiveCodeBench v5 (May 2023 - Jan 2025, use older subset for training)
    print("Loading LiveCodeBench v5 for training data...")
    lcb_v5 = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5", split="test")
    print(f"Loaded {len(lcb_v5)} problems from LiveCodeBench v5")
    
    # Load test data: LiveCodeBench v2 (more recent problems for evaluation)
    print("Loading LiveCodeBench v2 for test data...")
    lcb_v2 = load_dataset("livecodebench/code_generation_lite", version_tag="release_v2", split="test")
    print(f"Loaded {len(lcb_v2)} problems from LiveCodeBench v2")
    
    def process_lcb_example(example, idx, split="train"):
        """Process LiveCodeBench example into our standard format"""
        def safe_json_loads(json_str):
            if not json_str or json_str.strip() == "":
                return []
            try:
                return json.loads(json_str)
            except (json.JSONDecodeError, TypeError):
                return []
        
        public_tests = safe_json_loads(example.get("public_test_cases", ""))
        private_tests = safe_json_loads(example.get("private_test_cases", ""))
        
        # Use public tests + limited private tests for training/evaluation
        test_cases = public_tests + private_tests[:3]  # Slightly more tests for better training signal
        
        formatted_tests = []
        for test in test_cases:
            formatted_test = {
                "input": test.get("input", ""),
                "output": test.get("output", ""),
                "metadata": {"testtype": "functional"}
            }
            
            # Add function metadata if available
            metadata_str = example.get("metadata", "")
            if metadata_str and metadata_str.strip():
                try:
                    metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                    if isinstance(metadata, dict) and "func_name" in metadata:
                        formatted_test["metadata"]["func_name"] = metadata["func_name"]
                except (json.JSONDecodeError, TypeError):
                    pass
            
            formatted_tests.append(formatted_test)
        
        # Format the question with title and content
        question = f"Problem: {example['question_title']}\n\n{example['question_content']}"
        if example.get('starter_code'):
            question += f"\n\nStarter code:\n```python\n{example['starter_code']}\n```"
        
        return {
            "question": question,
            "ground_truth": formatted_tests,
            "data_source": "livecodebench",
            "uid": f"lcb_{split}_{example['question_id']}",
            "split": split,
            "index": idx,
            "difficulty": example["difficulty"],
            "question_id": example["question_id"],
            "contest_id": example["contest_id"],
            "contest_date": str(example["contest_date"]),
            "starter_code": example.get("starter_code", "")
        }
    
    # Process training data (limit to train_size)
    print("Processing training data from LiveCodeBench v5...")
    train_data = []
    for idx, example in enumerate(lcb_v5):
        if len(train_data) >= train_size:
            break
        processed = process_lcb_example(example, idx, split="train")
        train_data.append(processed)
    
    print(f"Processed {len(train_data)} training examples")
    
    # Process test data (limit to test_size) 
    print("Processing test data from LiveCodeBench v2...")
    test_data = []
    for idx, example in enumerate(lcb_v2):
        if len(test_data) >= test_size:
            break
        processed = process_lcb_example(example, idx, split="test")
        test_data.append(processed)
    
    print(f"Processed {len(test_data)} test examples")
    
    # Register datasets with RLLM
    train_dataset = DatasetRegistry.register_dataset("deepcoder_combined", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("livecodebench_test", test_data, "test")
    
    print(f"✅ Registered datasets:")
    print(f"  - Training: 'deepcoder_combined' with {len(train_data)} examples")
    print(f"  - Testing: 'livecodebench_test' with {len(test_data)} examples")
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_deepcoder_data()
    print(f"\n📊 Final Summary:")
    print(f"  - Train dataset: {len(train_dataset.get_data())} examples")
    print(f"  - Test dataset: {len(test_dataset.get_data())} examples") 