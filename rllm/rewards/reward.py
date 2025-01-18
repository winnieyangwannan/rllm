from enum import Enum


class RewardType(Enum):
    """
    Enum class for reward types
    """
    MATH = 'MATH'
    CODE = 'CODE'
    
    
class RewardMath(Enum):
    """
    Enum class for reward types
    """
    CORRECT = 1
    INCORRECT = -1
    FORMAT_ERROR = -1


class Reward:
    def __init__(self, reward_type: RewardType):
        self.reward_type = reward_type

    def __call__(self, response, label):
        raise NotImplementedError()