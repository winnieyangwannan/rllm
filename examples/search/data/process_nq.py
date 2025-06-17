#!/usr/bin/env python3
"""
Natural Questions dataset processing for Search-R1 training.
This script processes the NQ dataset into the format expected by the Search-R1 training pipeline.

Usage:
    python process_nq.py --input_dir ./search_data/nq --output_dir ./processed_data
"""

import argparse
import json
import os
from typing import Any

from datasets import load_dataset


def extract_answer_from_nq(example: dict[str, Any]) -> str:
    """Extract answer from Natural Questions example following Search-R1's approach."""
    annotations = example.get("annotations", {})

    # Try to get short answer first
    if "short_answers" in annotations and annotations["short_answers"]:
        short_answer = annotations["short_answers"][0]
        if "text" in short_answer:
            text = short_answer["text"]
            # Handle both string and list cases
            if isinstance(text, list):
                if text:  # non-empty list
                    return str(text[0]).strip()
            else:
                return str(text).strip()

    # Fall back to long answer
    if "long_answer" in annotations and annotations["long_answer"]:
        long_answer = annotations["long_answer"]
        if "candidate_text" in long_answer:
            # Take first 200 characters of long answer
            return str(long_answer["candidate_text"])[:200].strip()

    # Fall back to question if no answer found
    return "No answer found"


def process_nq_example(example: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    """Process a single NQ example into Search-R1 format."""

    # Extract question and answer
    question = example["question"]["text"].strip()
    answer = extract_answer_from_nq(example)

    # Create the data structure following Search-R1's format
    data = {
        "data_source": "natural_questions",
        "prompt": [
            {
                "role": "user",
                "content": f"Please answer the following question by searching for relevant information: {question}",
            }
        ],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"split": split, "index": idx, "task": {"question": question, "ground_truth": answer, "data_source": "natural_questions"}, "tools": ["local_search"], "uid": f"nq_{idx}", "question_type": "factual", "level": "easy"},
        "task": {"question": question, "ground_truth": answer, "data_source": "natural_questions"},
        "uid": f"nq_{idx}",
    }

    return data


def process_nq_dataset(input_dir: str = None, output_dir: str = "processed_data", train_size: int = 10000, val_size: int = 1000):
    """Process Natural Questions dataset following Search-R1's approach."""

    print("Processing Natural Questions dataset...")

    if input_dir and os.path.exists(input_dir):
        # Load from local files
        train_file = os.path.join(input_dir, "nq_train.json")
        val_file = os.path.join(input_dir, "nq_validation.json")

        if os.path.exists(train_file):
            with open(train_file) as f:
                train_dataset = json.load(f)
        else:
            train_dataset = []

        if os.path.exists(val_file):
            with open(val_file) as f:
                val_dataset = json.load(f)
        else:
            val_dataset = []

        print(f"Loaded {len(train_dataset)} train and {len(val_dataset)} validation examples from {input_dir}")
    else:
        # Load from HuggingFace
        try:
            dataset = load_dataset("natural_questions", trust_remote_code=True)
            train_dataset = list(dataset["train"])
            val_dataset = list(dataset["validation"])
            print(f"Loaded {len(train_dataset)} train and {len(val_dataset)} validation examples from HuggingFace")
        except Exception as e:
            print(f"Error loading dataset from HuggingFace: {e}")
            print("Please download the dataset manually first using download_search_data.py")
            return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process training data
    print(f"Processing training data (up to {train_size} examples)...")
    train_processed = []

    for idx, example in enumerate(train_dataset[:train_size]):
        processed = process_nq_example(example, idx, "train")
        train_processed.append(processed)

    # Process validation data
    print(f"Processing validation data (up to {val_size} examples)...")
    val_processed = []

    for idx, example in enumerate(val_dataset[:val_size]):
        processed = process_nq_example(example, idx, "test")
        val_processed.append(processed)

    # Save processed data
    train_output_file = os.path.join(output_dir, "nq_train_processed.json")
    val_output_file = os.path.join(output_dir, "nq_val_processed.json")

    with open(train_output_file, "w") as f:
        json.dump(train_processed, f, indent=2)

    with open(val_output_file, "w") as f:
        json.dump(val_processed, f, indent=2)

    print(f"Processed {len(train_processed)} training examples -> {train_output_file}")
    print(f"Processed {len(val_processed)} validation examples -> {val_output_file}")

    # Create combined file for training
    combined_file = os.path.join(output_dir, "nq_search_combined.json")
    combined_data = {"train": train_processed, "test": val_processed}

    with open(combined_file, "w") as f:
        json.dump(combined_data, f, indent=2)

    print(f"Combined dataset saved to {combined_file}")

    return train_output_file, val_output_file, combined_file


def main():
    parser = argparse.ArgumentParser(description="Process Natural Questions dataset for Search-R1 training")
    parser.add_argument("--input_dir", help="Input directory with NQ JSON files (optional, will download from HuggingFace if not provided)")
    parser.add_argument("--output_dir", default="processed_data", help="Output directory for processed data")
    parser.add_argument("--train_size", type=int, default=10000, help="Number of training examples to process")
    parser.add_argument("--val_size", type=int, default=1000, help="Number of validation examples to process")

    args = parser.parse_args()

    # Process the dataset
    train_file, val_file, combined_file = process_nq_dataset(input_dir=args.input_dir, output_dir=args.output_dir, train_size=args.train_size, val_size=args.val_size)

    print("\nNatural Questions processing completed!")
    print("Next steps:")
    print("1. Build the retrieval index: python retrieval/build_index.py")
    print("2. Launch the retrieval server: bash retrieval/launch_server.sh")
    print("3. Start PPO training: python train_search_agent.py")


if __name__ == "__main__":
    main()
