import enum
import json
from typing import Any, Dict, List, Union

class Datasets(enum.Enum):
    AIME = 'AIME'
    OMNI = 'OMNI'
    NUMINA = 'NUMINA'

def load_aime(path: str = "./raw/aime_v1.json") -> Union[List[Any], Dict[str, Any]]:
    """
    Loads the AIME dataset from the given JSON file and returns its contents.

    :param path: Path to the AIME JSON file.
    :return: The loaded AIME data (usually a list or dict, depending on the file).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_omni(path: str = "./raw/omni_math.json") -> Union[List[Any], Dict[str, Any]]:
    """
    Loads the OMNI (OMNIMath) dataset from the given JSON file and returns its contents.

    :param path: Path to the OMNI JSON file.
    :return: The loaded OMNI data (usually a list or dict, depending on the file).
    """
    with open(path, "r", encoding="utf-8") as f:
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

