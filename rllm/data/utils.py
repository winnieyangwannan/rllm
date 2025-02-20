import json
import os
from typing import Any, Dict, List

from rllm.data import Dataset, TrainDataset

def load_dataset(dataset: Dataset, local_dir=None) -> List[Dict[str, Any]]:
    """Loads a dataset from a JSON file.

    Args:
        dataset: The dataset to load.
        train: Whether to load the training or testing dataset.

    Returns:
        A list of dictionaries representing the dataset.

    Raises:
        ValueError: If the dataset is invalid or the file cannot be found.
    """

    dataset_name = dataset.value.lower()
    data_dir = "train" if isinstance(dataset, TrainDataset) else "test"
    
    # Get the current path of this file
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(data_dir, f"{dataset_name}.json")
    # Combine current_dir and file_path
    file_path = os.path.join(current_dir, file_path)
    
    if not os.path.exists(file_path):
        raise ValueError(f"Dataset file not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")
    except Exception as e: # Catch other potential file errors.
        raise ValueError(f"Error loading dataset: {e}")

if __name__ == '__main__':
    load_dataset(TrainDataset.NUMINA_OLYMPIAD)