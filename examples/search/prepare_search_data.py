from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_search_data(train_size=3000, test_size=100):
    """
    Prepare search datasets by loading HotpotQA and Natural Questions datasets,
    processing them, and registering them with the DatasetRegistry.
    
    Args:
        train_size: Maximum number of training examples to load
        test_size: Maximum number of test examples to load
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    print("Loading HotpotQA dataset...")
    hotpot_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)
    hotpot_train = hotpot_dataset["train"]
    hotpot_val = hotpot_dataset["validation"]
    
    print("Loading Natural Questions dataset...")
    nq_dataset = load_dataset("sentence-transformers/natural-questions", "pair")
    nq_train = nq_dataset["train"]
    
    # Select subsets
    hotpot_train_subset = hotpot_train.select(range(min(train_size // 2, len(hotpot_train))))
    hotpot_val_subset = hotpot_val.select(range(min(test_size // 2, len(hotpot_val))))
    nq_subset = nq_train.select(range(min(train_size // 2, len(nq_train))))
    
    def process_hotpot_example(example, idx, split):
        return {
            "question": example["question"],
            "ground_truth": example["answer"], 
            "data_source": "hotpotqa",
            "uid": f"hotpot_{example.get('id', idx)}",
            "split": split,
            "index": idx,
            "question_type": example.get("type", "bridge"),
            "level": example.get("level", "medium")
        }
    
    def process_nq_example(example, idx, split):
        ground_truth = example["answer"][:200] + "..." if len(example["answer"]) > 200 else example["answer"]
        return {
            "question": example["query"],
            "ground_truth": ground_truth,
            "data_source": "natural_questions",
            "uid": f"nq_{idx}",
            "split": split,
            "index": idx,
            "question_type": "factual",
            "level": "easy"
        }
    
    print("Processing HotpotQA training data...")
    hotpot_train_processed = [process_hotpot_example(example, idx, "train") for idx, example in enumerate(hotpot_train_subset)]
    
    print("Processing HotpotQA validation data...")
    hotpot_val_processed = [process_hotpot_example(example, idx, "test") for idx, example in enumerate(hotpot_val_subset)]
    
    print("Processing Natural Questions data...")
    nq_processed = [process_nq_example(example, idx, "train") for idx, example in enumerate(nq_subset)]
    
    # Combine datasets for training
    train_processed = hotpot_train_processed + nq_processed
    
    # Add remaining NQ examples to test set if needed
    remaining_nq_size = min(test_size - len(hotpot_val_processed), len(nq_train) - len(nq_subset))
    if remaining_nq_size > 0:
        nq_test_subset = nq_train.select(range(len(nq_subset), len(nq_subset) + remaining_nq_size))
        nq_test_processed = [process_nq_example(example, idx + len(nq_subset), "test") for idx, example in enumerate(nq_test_subset)]
        test_processed = hotpot_val_processed + nq_test_processed
    else:
        test_processed = hotpot_val_processed
    
    print(f"Combined dataset: {len(train_processed)} train examples, {len(test_processed)} test examples")
    
    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("search_combined", train_processed, "train")
    test_dataset = DatasetRegistry.register_dataset("search_combined", test_processed, "test")
    
    # Also register HotpotQA subset for compatibility
    hotpot_train_dataset = DatasetRegistry.register_dataset("hotpotqa_combined", hotpot_train_processed, "train")
    hotpot_test_dataset = DatasetRegistry.register_dataset("hotpotqa_combined", hotpot_val_processed, "test")
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_search_data()
    print(f"Train dataset: {train_dataset}")
    print(f"Test dataset: {test_dataset}") 