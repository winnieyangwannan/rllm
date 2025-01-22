from typing import List, Union, Dict

from torch.utils.data import Dataset as TorchDataset, DataLoader, WeightedRandomSampler, Sampler

from rllm.data.dataset_types import Dataset
from rllm.data.utils import load_dataset

DATASET_KEYS = ['problem', 'solution', 'answer']

def collate_fn(batch):
    """Collate function for the dataloader."""
    # Combine all items in the batch into a single dict
    return {
        key: [item[key] for item in batch]
        for key in DATASET_KEYS
    }

class WeightedDatasetSampler(Sampler):
    """Samples elements based on provided weights for each dataset."""
    
    def __init__(self, dataset_sizes: List[int], weights: List[float]):
        self.dataset_sizes = dataset_sizes
        self.weights = weights
        self.total_size = sum(dataset_sizes)
        self.offsets = [0] + [sum(dataset_sizes[:i+1]) for i in range(len(dataset_sizes)-1)]
        
        # Calculate per-sample weights
        sample_weights = []
        for _, (size, weight) in enumerate(zip(dataset_sizes, weights)):
            sample_weights.extend([weight / size] * size)
        self.sample_weights = sample_weights
        
    def __iter__(self):
        return iter(WeightedRandomSampler(self.sample_weights, len(self), replacement=True))
    
    def __len__(self):
        return self.total_size

class DatasetMix(TorchDataset):
    """A dataset that concatenates multiple datasets."""
    
    def __init__(
        self,
        datasets: Union[Dataset, List[Dataset], Dict[Dataset, float]],
    ):
        """Initialize DataMixture.
        
        Args:
            datasets: Union[Dataset, List[Dataset], Dict[Dataset, float]]
                Either a single Dataset, a list of Datasets, or a dictionary mapping 
                Datasets to their sampling weights (weights used to create sampler)
            batch_size: int
                Batch size for the dataloader
        """
        
        # Extract datasets and weights
        if isinstance(datasets, Dict):
            self.weights = list(datasets.values())
            dataset_enums = list(datasets.keys())
        else:
            dataset_enums = datasets if isinstance(datasets, list) else [datasets]
            self.weights = [1.0] * len(dataset_enums)
            
        # Load and concatenate datasets
        self.datasets = [load_dataset(d) for d in dataset_enums]
        self.dataset_sizes = [len(d) for d in self.datasets]
        
        # Calculate dataset offsets for indexing
        self.offsets = [0]
        for size in self.dataset_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + size)
        
    def __len__(self):
        return sum(self.dataset_sizes)
        
    def __getitem__(self, idx):
        """Get item by global index."""
        # Find which dataset this index belongs to
        dataset_idx = next(i for i, offset in enumerate(self.offsets + [len(self)]) 
                         if idx < offset) - 1
        if dataset_idx == -1:
            dataset_idx = 0
            
        # Get local index within dataset
        local_idx = idx - self.offsets[dataset_idx]
        sample = self.datasets[dataset_idx][local_idx]
        return sample


def make_dataloader(
    datasets: Union[Dataset, List[Dataset], Dict[Dataset, float]],
    batch_size: int = 8,
):
    """Create a DataLoader from a dataset or mixture of datasets.
    
    Args:
        datasets: Union[Dataset, List[Dataset], Dict[Dataset, float]]
            Either a single Dataset, a list of Datasets, or a dictionary mapping
            Datasets to their sampling weights
        batch_size: int
            Batch size for the dataloader
            
    Returns:
        DataLoader: DataLoader for the dataset(s)
    """
    dataset = DatasetMix(datasets)
    sampler = WeightedDatasetSampler(
        dataset_sizes=dataset.dataset_sizes,
        weights=dataset.weights
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn
    )

if __name__ == "__main__":
    from rllm.data.dataset_types import TrainDataset
    datasets = {
        TrainDataset.AIME: 0.5,
        TrainDataset.AMC: 0.5,
    }
    dataloader = make_dataloader(datasets, batch_size=8)
    for batch in dataloader:
        print(batch)
        import pdb; pdb.set_trace()
        break
