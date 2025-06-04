import numpy as np
import os
import sys
from datasets import load_dataset

# Add the parent directory to sys.path to import from rllm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rllm.data.dataset import Dataset, DatasetRegistry

def generate_search_r1_data(train_size=100, test_size=20):
    """Generate sample Search-R1 data by combining HotpotQA and Natural Questions datasets."""
    print("Loading HotpotQA dataset...")
    
    # Load HotpotQA dataset (using distractor subset)
    hotpot_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)
    hotpot_train = hotpot_dataset["train"]
    hotpot_val = hotpot_dataset["validation"]
    
    print("Loading Natural Questions dataset...")
    
    # Load Natural Questions dataset 
    nq_dataset = load_dataset("sentence-transformers/natural-questions", "pair")
    nq_train = nq_dataset["train"]
    
    # Take subsets for testing
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
            "ground_truth": example["answer"][:200] + "..." if len(example["answer"]) > 200 else example["answer"],  # Truncate long answers
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
    print(f"Training sources: {len(hotpot_train_processed)} HotpotQA + {len(nq_processed)} Natural Questions")
    print(f"Test sources: {len(hotpot_val_processed)} HotpotQA + {len(test_processed) - len(hotpot_val_processed)} Natural Questions")
    
    return train_processed, test_processed

def main():
    # Generate combined dataset
    train_data, test_data = generate_search_r1_data()
    print(f"Generated {len(train_data)} train examples and {len(test_data)} test examples")

    dataset_name = "search_r1_combined"
    
    # Check if train split already exists
    if DatasetRegistry.dataset_exists(dataset_name, "train"):
        print(f"Dataset '{dataset_name}' train split already exists, loading it...")
        train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
        print(f"Loaded train split with {len(train_dataset)} examples")
    else:
        # Register the train split
        print(f"Registering new dataset '{dataset_name}' train split...")
        train_dataset = DatasetRegistry.register_dataset(dataset_name, train_data, "train")
        print(f"Registered train split with {len(train_dataset)} examples")

    # Check if test split already exists
    if DatasetRegistry.dataset_exists(dataset_name, "test"):
        print(f"Dataset '{dataset_name}' test split already exists, loading it...")
        test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")
        print(f"Loaded test split with {len(test_dataset)} examples")
    else:
        # Register the test split
        print(f"Registering new dataset '{dataset_name}' test split...")
        test_dataset = DatasetRegistry.register_dataset(dataset_name, test_data, "test")
        print(f"Registered test split with {len(test_dataset)} examples")

    # Get all dataset names
    dataset_names = DatasetRegistry.get_dataset_names()
    print(f"Available datasets: {dataset_names}")
    
    # Get all splits for the dataset
    splits = DatasetRegistry.get_dataset_splits(dataset_name)
    print(f"Available splits for '{dataset_name}': {splits}")

    # Sample some data from train split
    print("\nSample train data:")
    for i in range(min(3, len(train_dataset))):
        example = train_dataset[i]
        print(f"Question: {example['question']}")
        print(f"Ground Truth: {example['ground_truth']}")
        print(f"Data Source: {example['data_source']}")
        print(f"Type: {example.get('question_type', 'N/A')}")
        print("---")

    # Sample some data from test split
    print("\nSample test data:")
    for i in range(min(3, len(test_dataset))):
        example = test_dataset[i]
        print(f"Question: {example['question']}")
        print(f"Ground Truth: {example['ground_truth']}")
        print(f"Data Source: {example['data_source']}")
        print(f"Type: {example.get('question_type', 'N/A')}")
        print("---")

if __name__ == "__main__":
    main() 