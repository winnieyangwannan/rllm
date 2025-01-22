from typing import List

from rllm.data import Problem
from rllm.sampler import SampleBatch
from rllm.rewards import RewardType, RewardInput, RewardOutput

def convert_batch_to_reward_input(problem: Problem, sample_batch: SampleBatch) -> RewardInput:
    reward_inputs = []
    for s in sample_batch.samples:
        reward_inputs.append(RewardInput(
            problem=problem.problem,
            problem_type=RewardType.MATH,
            model_response=s.response,
            metadata={
                'answer': problem.answer,
            },
        ))
    return reward_inputs

def convert_reward_output_to_batch(sample_batch: SampleBatch, reward_outputs: List[RewardOutput]) -> SampleBatch:
    for i, s in enumerate(sample_batch.samples):
        s.reward = reward_outputs[i].reward
        s.is_correct = reward_outputs[i].is_correct
    return sample_batch
 