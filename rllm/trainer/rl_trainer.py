from rllm.data import TrainDataset, make_dataloader
from rllm.sampler import DistributedSampler
from rllm.rewards.math_reward import RewardMathFn
from rllm.worker.rollout_worker import RolloutWorker
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class RLTrainerConfig:
    # RL Config
    algorithm: str = "reinforce"
    # Batch size for RL training.
    train_batch_size: int = 8
    # Sampler Config.
    # Number of samples per problem.
    samples_per_problem: int = 4
    # Number of workers for distributed sampling.
    num_workers: int = 1
    # Tensor parallel size for each sampler worker.
    tensor_parallel_size: int = 2
    # Sampler backend. Either SGLang or VLLM.
    sampler_backend: str = "SGLang"
    # Model for distributed sampling.
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    
    # Reward Config.
    # blah blah

if __name__ == "__main__":
    batch_size = 1
    datasets = {
        TrainDataset.AIME: 0.5,
        TrainDataset.AMC: 0.5, 
    }
    dataloader = make_dataloader(datasets, batch_size=batch_size)
    
    reward_fn = RewardMathFn()
    
    sampler = DistributedSampler(
        backend="vllm",
        num_workers=1,
        tensor_parallel_size=2,
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    )
    sampler_workers = [RolloutWorker(sampler, reward_fn) for _ in range(batch_size)]
    import pdb; pdb.set_trace()
    # Get next batch from dataloader.
    for batch in dataloader:
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = []
            for i, problem in enumerate(batch):
                futures.append(executor.submit(sampler_workers[i].rollout, problem))
            sample_batch_outputs = [future.result() for i, future in enumerate(futures)]
        break
    import pdb; pdb.set_trace()
    # Shutdown sampler to launch trainer lol.
    sampler.shutdown()
    # TODO: Launch and implement trainer.