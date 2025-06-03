"""Postprocessing functions for different dataset types.

This module contains reusable postprocessing functions that can be properly
serialized and imported for the persistent dataset registry system.
"""

import json
from typing import Dict, Any


def process_lcb(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Process LiveCodeBench dataset."""
    return {
        "problem": example.get("problem", ""),
        "tests": example.get("tests", []),
        "data_source": "livecodebench",
        "metadata": example.get("metadata", {})
    }


def process_apps(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Process APPS dataset."""
    question = example.get("problem", "")
    return {
        "ground_truth": json.dumps(example.get("input_output", {})),
        "question": f"Write a solution for this programming problem:\n\n{question}",
        "idx": idx,
        "data_source": "apps"
    }


def process_mbpp(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Process MBPP dataset."""
    question = example.get("text", "")
    return {
        "ground_truth": json.dumps(example.get("test_cases", [])),
        "question": f"Write a Python function to solve:\n\n{question}",
        "idx": idx,
        "data_source": "mbpp"
    }


def process_math_dataset(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Process math datasets (GSM8K, etc.)."""
    question = example.get("question", "") or example.get("problem", "")
    answer = example.get("answer", "") or example.get("solution", "")
    
    return {
        "ground_truth": answer,
        "question": question,
        "idx": idx,
        "data_source": "math"
    }


def process_code_dataset(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Standard processing for code datasets."""
    return {
        "data_source": "code",
        "prompt": [{
            "role": "user", 
            "content": example.get("problem", "")
        }],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(example.get("tests", []))
        },
        "extra_info": {
            "split": "train",
            "index": idx,
            "task": {
                "data_source": "code",
                "question": example.get("problem", ""),
                "ground_truth": json.dumps(example.get("tests", []))
            }
        }
    }


def process_hellaswag(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Process HellaSwag dataset."""
    return {
        "context": example.get("ctx", ""),
        "endings": example.get("endings", []),
        "label": example.get("label", ""),
        "idx": idx,
        "data_source": "hellaswag"
    }


# Registry of available processors for easy lookup
PROCESSORS = {
    "lcb": process_lcb,
    "livecodebench": process_lcb,
    "apps": process_apps,
    "mbpp": process_mbpp,
    "math": process_math_dataset,
    "gsm8k": process_math_dataset,
    "code": process_code_dataset,
    "hellaswag": process_hellaswag,
} 