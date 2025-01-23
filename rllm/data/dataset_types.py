from dataclasses import dataclass
import enum
from typing import Any, Dict, List, Union

class TrainDataset(enum.Enum):
    # The standard American beginner competitions.
    AIME = 'AIME'
    AMC =  'AMC'
    # Omni math dataset
    OMNI_MATH = 'OMNI_MATH'
    # Unique Olympiad problems from NUMINA
    NUMINA_OLYMPIAD = 'OLYMPIAD'
    # Dan Hendrycks math
    MATH = 'MATH'

class TestDataset(enum.Enum):
    AIME = 'AIME'
    AMC =  'AMC'
    MATH = 'MATH500'

Dataset = Union[TrainDataset, TestDataset]

@dataclass
class Problem:
    problem: str
    solution: str 
    answer: str
    difficulty: float
    dataset: Dataset