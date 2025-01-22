from rllm.data import TrainDataset, make_dataloader
from rllm.sampler import DistributedSGLang


if __name__ == "__main__":
    datasets = {
        TrainDataset.AIME: 0.5,
        TrainDataset.AMC: 0.5, 
    }
    dataloader = make_dataloader(datasets, batch_size=8)
    for batch in dataloader:
        print(batch)
        break
    
    sampler = DistributedSGLang(
        num_workers=2,
        tensor_parallel_size=2,
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    )
    sampler.shutdown()