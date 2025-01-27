from typing import List
from concurrent.futures import ThreadPoolExecutor

from rllm.data import Problem
from rllm.sampler import SampleBatch, DistributedSampler, SampleConfig
from rllm.system_prompts import DEEPSEEK_MATH_SYSTEM_PROMPT
from rllm.rewards import RewardFn
from rllm.worker.utils import convert_batch_to_reward_input, convert_reward_output_to_batch

class RolloutWorker:
    def __init__(self, config: SampleConfig, sampler: DistributedSampler, reward_fn: RewardFn):
        self.sampler = sampler
        self.reward_fn = reward_fn
        self.config = config

    def rollout(self, problem: Problem) -> SampleBatch:
        extracted_problem = problem.problem
        messages = [
            {"role": "user", "content": f"{DEEPSEEK_MATH_SYSTEM_PROMPT} Problem: {extracted_problem}"},
        ]
        # Fetch sampled batch.
        sample_batch: SampleBatch = self.sampler.chat_completion(messages=messages,
                                                          temperature=self.config.temperature,
                                                          n=self.config.samples_per_problem,
                                                          max_tokens=self.config.max_tokens)
        
        reward_inputs = convert_batch_to_reward_input(problem, sample_batch)
        with ThreadPoolExecutor(max_workers=len(reward_inputs)) as executor:
            reward_outputs = list(executor.map(lambda x: self.reward_fn(x), reward_inputs))
        
        processed_batch = convert_reward_output_to_batch(sample_batch, reward_outputs)
        return processed_batch
        
        
        
