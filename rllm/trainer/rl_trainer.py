from concurrent.futures import ThreadPoolExecutor

from rllm.config import RLTrainerConfig
from rllm.data import DataLoaderFn
from rllm.rewards.rl_reward import RLRewardFn
from rllm.sampler import DistributedSampler
from rllm.worker.rollout_worker import RolloutWorker

def main_rl_train_loop(config: RLTrainerConfig):
    # Create dataloader using config
    dataset_config = config.get_dataset_config()
    dataloader = DataLoaderFn(dataset_config)
    
    # Initialize reward function using config
    reward_config = config.get_reward_config()
    reward_fn = RLRewardFn(reward_config)
    # Create sampler using config parameters
    sample_config = config.get_sample_config()
    sampler = DistributedSampler(
        backend=sample_config.sampler_backend,
        num_workers=sample_config.num_workers,
        tensor_parallel_size=sample_config.tensor_parallel_size,
        model=sample_config.model,
    )

    problems_per_batch  = config.train_batch_size // config.samples_per_problem
    # Create workers based on batch size
    sampler_workers = [
        RolloutWorker(sample_config, sampler, reward_fn) 
        for _ in range(problems_per_batch)
    ]

    # Get next batch from dataloader
    for batch in dataloader:
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = []
            for i, problem in enumerate(batch):
                futures.append(executor.submit(sampler_workers[i].rollout, problem))
            sample_batch_outputs = [future.result() for future in futures]
        break
    import pdb; pdb.set_trace()
    # Cleanup
    sampler.shutdown()
    # TODO: Launch and implement trainer.
    
if __name__ == "__main__":
    config = RLTrainerConfig()
    main_rl_train_loop(config)