"""Example demonstrating how to use the Dataset class for loading and processing data."""

import json
import os
from typing import Dict, List

from rllm.data.dataset import Dataset, DatasetRegistry
from rllm.data.dataset_types import TrainDataset, TestDataset


def example_math_data_loading():
    """Example of loading math datasets."""
    # Load GSM8K directly from HuggingFace
    gsm8k_dataset = Dataset(
        dataset_name="GSM8K",
        split="test",
        load_from_hf=True
    )
    print(f"Loaded {len(gsm8k_dataset)} GSM8K test problems")
    print(f"First example: {gsm8k_dataset[0]}")
    
    # Load MATH500 from HuggingFace with custom processing
    def custom_math_process(example, idx):
        # Example of a custom processing function
        question = example.get("problem", "")
        instruction = "Solve this math problem. Show all your work and explain each step."
        task = {
            "ground_truth": example.get("answer", ""),
            "question": f"{question}\n\n{instruction}",
            "idx": idx,
            "data_source": "custom_math"
        }
        return task
    
    math500_dataset = Dataset(
        dataset_name="MATH500",
        split="test", 
        load_from_hf=True,
        trust_remote_code=True,
        postprocess_fn=custom_math_process
    )
    print(f"Loaded {len(math500_dataset)} MATH500 test problems with custom processing")
    print(f"First example: {math500_dataset[0]}")
    
    # Load from local files (fallback)
    aime_dataset = Dataset(
        dataset_name=TestDataset.Math.AIME,
        split="test",
        load_from_hf=False  # Force local loading
    )
    print(f"Loaded {len(aime_dataset)} AIME test problems from local files")
    print(f"First example: {aime_dataset[0]}")


def example_code_data_loading():
    """Example of loading code datasets."""
    # Load LiveCodeBench directly from HuggingFace
    lcb_dataset = Dataset(
        dataset_name="LIVECODEBENCH",
        split="test",
        load_from_hf=True
    )
    print(f"Loaded {len(lcb_dataset)} LiveCodeBench test problems")
    print(f"First example: {lcb_dataset[0]}")
    
    # Load APPS with custom processing
    def custom_code_process(example, idx):
        # Example of a custom processing function for code problems
        question = example.get("problem", "")
        task = {
            "ground_truth": json.dumps(example.get("test_cases", [])),
            "question": f"Write a solution for this programming problem:\n\n{question}",
            "idx": idx,
            "data_source": "custom_code"
        }
        return task
    
    apps_dataset = Dataset(
        dataset_name="APPS",
        split="test",
        load_from_hf=True,
        postprocess_fn=custom_code_process
    )
    print(f"Loaded {len(apps_dataset)} APPS test problems with custom processing")
    print(f"First example: {apps_dataset[0]}")


def example_custom_dataset_registration():
    """Example of registering and using custom datasets."""
    # Register a custom dataset with HuggingFace name and postprocessing function
    @DatasetRegistry.register_dataset(dataset_name="MBPP", hf_dataset_name="mbpp")
    def process_mbpp(example, idx):
        """Process MBPP (Mostly Basic Python Programming) dataset."""
        question = example.get("text", "")
        task = {
            "ground_truth": json.dumps(example.get("test_cases", [])),
            "question": f"Write a Python function to solve:\n\n{question}",
            "idx": idx,
            "data_source": "mbpp"
        }
        return task
    
    # Register custom dataset with only processing function (no HF name)
    def process_custom_json(example, idx):
        """Process a custom JSON dataset."""
        return {
            "ground_truth": example.get("answer", ""),
            "question": example.get("question", ""),
            "idx": idx,
            "data_source": "custom_json"
        }
    
    DatasetRegistry.register_dataset(
        dataset_name="CUSTOM_JSON", 
        hf_dataset_name=None,  # No HuggingFace equivalent
        postprocess_fn=process_custom_json
    )
    
    # Register a dataset with just HuggingFace name, no custom processing
    DatasetRegistry.register_dataset(
        dataset_name="HELLASWAG", 
        hf_dataset_name="hellaswag"
    )
    
    # Use the registered custom dataset
    try:
        mbpp_dataset = Dataset(
            dataset_name="MBPP",
            split="test",
            load_from_hf=True
        )
        print(f"Loaded {len(mbpp_dataset)} MBPP test problems")
        print(f"First example: {mbpp_dataset[0]}")
    except Exception as e:
        print(f"Failed to load MBPP: {e}")
    
    # Print all registered datasets
    print(f"All registered datasets: {DatasetRegistry.list_registered_datasets()}")


def save_processed_dataset(dataset, output_path):
    """Save a processed dataset to a JSON file.
    
    Args:
        dataset: The dataset to save
        output_path: Path to save the dataset to
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dataset.data, f, indent=2)
    print(f"Saved {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    print("=== Math Dataset Examples ===")
    example_math_data_loading()
    
    print("\n=== Code Dataset Examples ===")
    example_code_data_loading()
    
    print("\n=== Custom Dataset Registration Examples ===")
    example_custom_dataset_registration()
    
    # Example of saving a processed dataset
    gsm8k_dataset = Dataset("GSM8K", split="test", load_from_hf=True)
    save_processed_dataset(gsm8k_dataset, "processed_data/gsm8k_processed.json") 