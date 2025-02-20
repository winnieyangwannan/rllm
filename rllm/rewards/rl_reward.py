from rllm.rewards.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from rllm.rewards.math_reward import RewardMathFn

# Check RewardConfig to understand the config values.
class RLRewardFn(RewardFn):
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.math_reward_fn = RewardMathFn(config)
        self.cot_reward_fn = None

    def __call__(self, input: RewardInput) -> RewardOutput:
        reward_type = input.problem_type
        reward = 0
        is_correct = False
        if reward_type == RewardType.MATH:
            math_reward_output = self.math_reward_fn(input)
            reward += self.config.math_reward_weight * math_reward_output.reward
            is_correct = math_reward_output.is_correct
        elif reward_type == RewardType.CODE:
            pass
        else:
            raise ValueError(f"Unsupported reward type: {reward_type}")
        
        if self.config.cot_reward_weight != 0:
            cot_reward_output = self.cot_reward_fn(input)
            reward += self.config.cot_reward_weight * cot_reward_output.reward
        
        return RewardOutput(
            reward=reward,
            is_correct=is_correct
        )
