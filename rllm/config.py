from rllm.data import Dataset, DatasetConfig
from rllm.sampler import SampleConfig
from rllm.rewards.reward_types import RewardConfig
from dataclasses import dataclass, field, fields
from typing import List, Union

@dataclass
class RLTrainerConfig:
    """Configuration class for RL training.
    
    Contains settings for training algorithm, sampling, and reward calculation.
    
    Training Configuration:
        algorithm (str): RL algorithm to use (e.g. "reinforce")
        train_batch_size (int): Batch size for training updates
        
    Dataset Configuration:
        datasets (List[Dataset]): List of datasets to use for training (only TrainDataset is allowed!)
        dataset_weights (List[float]): Weights for sampling from each dataset

    Sampling Configuration:
        samples_per_problem (int): Number of samples to generate per input problem
        num_workers (int): Number of parallel sampling workers
        tensor_parallel_size (int): Tensor parallelism degree per worker
        sampler_backend (str): Backend to use for sampling ("SGLang" or "VLLM")
        model (str): Model path/name for distributed sampling
        temperature (float): Sampling temperature for generation
        
    Reward Configuration:
        math_reward_weight (float): Weight for math-based rewards
        use_math_orm (bool): Whether to use LLM oracle for math evaluation
        cot_reward_weight (float): Weight for chain-of-thought penalty.
        correct_reward (float): Reward value for correct answers
        incorrect_reward (float): Reward value for incorrect answers  
        format_error_reward (float): Reward value for formatting errors
        unk_error_reward (float): Reward value for unknown/error cases
    """
    # Training Configuration
    algorithm: str = "reinforce" # TODO: Add more algorithms, rn either reinforce or grpo.
    train_batch_size: int = 32 # Number of problems sampled is train_batch_size // samples_per_problem
    
    # Dataset Configuration
    datasets: Union[List[Dataset], List[str], str] = field(default_factory=lambda: ["AMC", "AIME"])
    dataset_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    
    # Sampling Configuration  (see SampleConfig for more details)
    samples_per_problem: int = 4
    num_workers: int = 2
    tensor_parallel_size: int = 2
    sampler_backend: str = "SGLang"
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    temperature: float = 0.6
    max_tokens: int = 8096
    
    # Reward Configuration (see RewardConfig for more details)
    math_reward_weight: float = 1.0
    use_math_orm: bool = True
    cot_reward_weight: float = 0.0
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0
    format_error_reward: float = -1.0
    unk_error_reward: float = -1.0

    def get_reward_config(self) -> RewardConfig:
        """Build RewardConfig from the RLTrainerConfig."""
        reward_fields = {f.name for f in fields(RewardConfig)}
        reward_args = {
            field: getattr(self, field)
            for field in reward_fields
            if hasattr(self, field)
        }
        return RewardConfig(**reward_args)

    def get_dataset_config(self) -> DatasetConfig:
        """Build DatasetConfig from the RLTrainerConfig."""
        dataset_fields = {f.name for f in fields(DatasetConfig)}
        dataset_args = {
            field: getattr(self, field)
            for field in dataset_fields
            if hasattr(self, field)
        }
        dataset_args['dataloader_batch_size'] = self.train_batch_size // self.samples_per_problem
        return DatasetConfig(**dataset_args)

    def get_sample_config(self) -> SampleConfig:
        """Build SampleConfig from the RLTrainerConfig."""
        sample_fields = {f.name for f in fields(SampleConfig)}
        sample_args = {
            field: getattr(self, field)
            for field in sample_fields
            if hasattr(self, field)
        }
        return SampleConfig(**sample_args)