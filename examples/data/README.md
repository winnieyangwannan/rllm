# Unified Data Interface Examples

This directory contains examples of using the unified data interface for loading and processing datasets. The interface provides a simple way to load data from different sources and process it into a standardized format for use in inference.

## Dataset Class

The `Dataset` class is a PyTorch-compatible dataset that:

1. Loads data directly from Hugging Face based on the dataset name
2. Postprocesses data into the desired format for inference
3. Falls back to loading from local files if Hugging Face loading fails

## Dataset Registry

The `DatasetRegistry` provides a way to register custom datasets and postprocessing functions. This allows users to:

1. Register their own datasets with custom Hugging Face names
2. Register custom postprocessing functions for datasets
3. Use registered datasets seamlessly with the Dataset class

## Usage Examples

The `load_datasets.py` file shows several examples of using the Dataset class:

- Loading math datasets (GSM8K, MATH500, AIME)
- Loading code datasets (LiveCodeBench, APPS)
- Using custom processing functions
- Registering and using custom datasets
- Saving processed datasets to JSON files

## Basic Usage

```python
from rllm.data.dataset import Dataset

# Load data directly from Hugging Face
dataset = Dataset(
    dataset_name="GSM8K",
    split="test",
    load_from_hf=True
)

# Use custom processing
def custom_process(example, idx):
    # Process the example
    return processed_example

dataset = Dataset(
    dataset_name="MATH500",
    split="test",
    load_from_hf=True,
    postprocess_fn=custom_process
)

# Load data from local files
from rllm.data.dataset_types import TestDataset
dataset = Dataset(
    dataset_name=TestDataset.Math.AIME,
    split="test",
    load_from_hf=False
)
```

## Registering Custom Datasets

```python
from rllm.data.dataset import DatasetRegistry, Dataset

# Method 1: Register using decorator
@DatasetRegistry.register_dataset(dataset_name="MBPP", hf_dataset_name="mbpp")
def process_mbpp(example, idx):
    """Process MBPP dataset."""
    # Process the example
    return processed_example

# Method 2: Register explicitly
def process_custom(example, idx):
    """Process custom dataset."""
    # Process the example
    return processed_example

DatasetRegistry.register_dataset(
    dataset_name="CUSTOM", 
    hf_dataset_name="my-org/custom-dataset",
    postprocess_fn=process_custom
)

# Use registered datasets
dataset = Dataset(dataset_name="MBPP", split="test")
```

## Data Processing

The Dataset class includes default processing functions for math and code datasets:

- `process_math_fn`: Processes math problems with step-by-step instructions
- `process_code_fn`: Processes code problems with LiveCodeBench formatting

These functions can be overridden by providing a custom `postprocess_fn` when creating the dataset or by registering a custom postprocessing function in the registry.

## Adding New Datasets

To add support for new datasets:

1. Register the dataset with `DatasetRegistry`:
   ```python
   DatasetRegistry.register_dataset(
       dataset_name="MY_DATASET",
       hf_dataset_name="org/dataset-name",
       postprocess_fn=my_process_fn
   )
   ```

2. For permanent additions, add the dataset enum to `TrainDataset` or `TestDataset` in `rllm/data/dataset_types.py`

3. The dataset can then be used with:
   ```python
   dataset = Dataset(dataset_name="MY_DATASET", split="train")
   ``` 