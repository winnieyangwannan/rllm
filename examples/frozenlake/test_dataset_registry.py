import numpy as np
import os
import sys

# Add the parent directory to sys.path to import from rllm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rllm.data.dataset import Dataset, DatasetRegistry

def generate_frozenlake_data(train_size=100, test_size=20):
    """Generate sample FrozenLake data for testing the dataset registry."""
    np.random.seed(42)
    train_seeds = np.random.randint(0, 100000, size=train_size)
    test_seeds = np.random.randint(0, 100000, size=test_size)
    train_sizes = np.random.randint(2, 10, size=train_size)
    test_sizes = np.random.randint(2, 10, size=test_size)
    train_ps = np.random.uniform(0.6, 0.85, size=train_size)
    test_ps = np.random.uniform(0.6, 0.85, size=test_size)

    def frozenlake_process_fn(seed, size, p, idx):
        return {
            "seed": seed,
            "size": size,
            "p": p,
            "index": idx,
            "uid": f"{seed}_{size}_{p}"
        }

    train_data = [frozenlake_process_fn(seed, train_sizes[idx], train_ps[idx], idx) for idx, seed in enumerate(train_seeds)]
    test_data = [frozenlake_process_fn(seed, test_sizes[idx], test_ps[idx], idx) for idx, seed in enumerate(test_seeds)]

    return train_data, test_data

def main():
    # Generate sample data
    train_data, test_data = generate_frozenlake_data()
    print(f"Generated {len(train_data)} train examples and {len(test_data)} test examples")

    dataset_name = "frozenlake_test"
    
    # Check if train split already exists
    if DatasetRegistry.dataset_exists(dataset_name, "train"):
        print(f"Dataset '{dataset_name}' train split already exists, loading it...")
        train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
        print(f"Loaded train split with {len(train_dataset)} examples")
    else:
        # Register the train split
        print(f"Registering new dataset '{dataset_name}' train split...")
        train_dataset = DatasetRegistry.register_dataset(dataset_name, train_data, "train")
        print(f"Registered train split with {len(train_dataset)} examples")

    # Check if test split already exists
    if DatasetRegistry.dataset_exists(dataset_name, "test"):
        print(f"Dataset '{dataset_name}' test split already exists, loading it...")
        test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")
        print(f"Loaded test split with {len(test_dataset)} examples")
    else:
        # Register the test split
        print(f"Registering new dataset '{dataset_name}' test split...")
        test_dataset = DatasetRegistry.register_dataset(dataset_name, test_data, "test")
        print(f"Registered test split with {len(test_dataset)} examples")

    # Get all dataset names
    dataset_names = DatasetRegistry.get_dataset_names()
    print(f"Available datasets: {dataset_names}")
    
    # Get all splits for the dataset
    splits = DatasetRegistry.get_dataset_splits(dataset_name)
    print(f"Available splits for '{dataset_name}': {splits}")

    # Sample some data from train split
    print("\nSample train data:")
    for i in range(min(3, len(train_dataset))):
        print(train_dataset[i])

    # Sample some data from test split
    print("\nSample test data:")
    for i in range(min(3, len(test_dataset))):
        print(test_dataset[i])

    # Optional: Remove specific splits or the entire dataset (uncomment to test)
    # if input("Do you want to remove the test split? (y/n): ").lower() == 'y':
    #     removed = DatasetRegistry.remove_dataset_split(dataset_name, "test")
    #     print(f"Test split removed: {removed}")
    #     splits = DatasetRegistry.get_dataset_splits(dataset_name)
    #     print(f"Available splits after removal: {splits}")
    #
    # if input("Do you want to remove the entire dataset? (y/n): ").lower() == 'y':
    #     removed = DatasetRegistry.remove_dataset(dataset_name)
    #     print(f"Dataset removed: {removed}")
    #     dataset_names = DatasetRegistry.get_dataset_names()
    #     print(f"Available datasets after removal: {dataset_names}")

if __name__ == "__main__":
    main() 