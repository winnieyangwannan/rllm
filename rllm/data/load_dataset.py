import enum
import json
import os
from typing import Any, Dict, List, Union

class Datasets(enum.Enum):
    AIME = 'AIME'
    OMNI = 'OMNI'
    NUMINA = 'NUMINA'

class DatasetType(enum.Enum):
    TRAIN = 'TRAIN'
    TEST = 'TEST'

def load_aime(dataset_type = DatasetType.TRAIN) -> Union[List[Any], Dict[str, Any]]:
    """
    Loads the AIME dataset from the given JSON file and returns its contents.

    Convert AIME dataset into universal format ['problem', 'solution', 'answer', 'difficulty']

    :param path: Path to the AIME JSON file.
    :return: The loaded AIME data (usually a list or dict, depending on the file).
    """
    subfolder_name = 'train'
    if dataset_type == DatasetType.TEST:
        subfolder_name = 'test'
    # Get the current directory path of this file
    full_path = os.path.dirname(os.path.abspath(__file__)) + f'/raw/{subfolder_name}/aime.json'
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_omni() -> Union[List[Any], Dict[str, Any]]:
    """
    Loads the OMNI (OMNIMath) dataset from the given JSON file and returns its contents.

    :param path: Path to the OMNI JSON file.
    :return: The loaded OMNI data (usually a list or dict, depending on the file).
    """
    full_path = os.path.dirname(os.path.abspath(__file__)) + f'/raw/train/omni_math.jsonl'
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_dataset(dataset: Datasets):
    if dataset == Datasets.AIME:
        return load_aime()
    elif dataset == Datasets.OMNI:
        return load_omni()
    elif dataset == Datasets.NUMINA:
        pass
        #return load_numina()
    raise ValueError('Invalid dataset')
