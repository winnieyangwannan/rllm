"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
import json 

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

from rllm.data.utils import load_dataset
from rllm.data.dataset_types import TrainDataset, TestDataset
from rllm.data.dataloader import DatasetMix 

def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int, dataset_name=None) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        question = f"{question} {instruction}"
        if dataset_name == "MATH":
            answer = example.pop('answer')#str
        elif dataset_name== "taco" or dataset_name == "apps": #taco/apps code datasets
            answer = dict()
            answer["input_output"] = example.pop('input_output') #dict
            answer = json.dumps(answer)
        elif dataset_name == "codeforces":
            answer = dict()
            answer["test_cases"] = example.pop('test_cases')
            answer = json.dumps(answer)
        elif dataset_name == "code_contests":
            answer = dict()
            answer["public_tests"] = example.pop('public_tests')
            answer = json.dumps(answer)
        elif dataset_name == "livecodebench":
            answer = dict()
            answer["public_test_cases"] = example.pop('public_test_cases')
            answer = json.dumps(answer)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        data = {
            "data_source": dataset_name,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer#set for the different dataset 
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn


#python3 code_dataset.py --local_dir /data/xiaoxiang/data/

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default=os.path.expanduser('~/rllm/data'),
                       help='Local directory to save processed datasets')#Xiao:hardcode,need to change 
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # Make local directory if it doesn't exist
    if not os.path.exists(local_dir):
        makedirs(local_dir)

    # Initialize datasets
    train_datasets = [TrainDataset.TACO, TrainDataset.APPS,TrainDataset.CODE_CONTESTS]
    train_dataset_names = ["taco", "apps", "code_contests"]
    print(f"1 scripts/data/deepscaler_dataset.py ,the train_datasets[0] is {train_datasets[0]}")
    test_datasets = [TestDataset.TACO, TestDataset.APPS, TestDataset.CODE_CONTESTS]
    test_datasets_names = ["taco", "apps","code_contests"]

    # #test_datasets = [TestDataset.AIME, TestDataset.AMC, TestDataset.MATH, TestDataset.MINERVA, TestDataset.OLYMPIAD_BENCH]
    # test_datasets = [TestDataset.LIVECODEBENCH]
    
    test_datasets_data = [load_dataset(d, local_dir) for d in test_datasets]
    train_dataset_data = [load_dataset(d, local_dir) for d in train_datasets]
    
    # Process training data
    all_train_data = [] 
    process_fn = make_map_fn('train')

    for train_dataset, train_dataset_name in zip(train_dataset_data, train_dataset_names):
        train_data: List[Dict[str, Any]] = []
        for idx, example in enumerate(train_dataset):
            processed_example = process_fn(example, idx, train_dataset_name)

            if processed_example is not None:
                train_data.append(processed_example)
                all_train_data.append(processed_example)
        train_data = train_data[:5000]#TODO(xiao):if we use parquet, the dataset size can not be too large, otherwise, it can not read
        train_df = pd.DataFrame(train_data)
        train_df.to_parquet(os.path.join(local_dir, f'train_{train_dataset_name}.parquet'))#train parquet for each code dataset
    
    #save all code dataset
    all_train_df = pd.DataFrame(all_train_data)
    all_train_df.to_parquet(os.path.join(local_dir, 'train_code.parquet')) #train parquet for all code dataset

    #Process and save each test dataset separately
    all_test_data = []
    for test_dataset, test_data_list, test_datasets_name in zip(test_datasets, test_datasets_data, test_datasets_names):
        test_data: List[Dict[str, Any]] = []
        process_fn = make_map_fn('test')
        for idx, example in enumerate(test_data_list):
            processed_example = process_fn(example, idx, test_datasets_name)
            groud_truth = processed_example['reward_model']['ground_truth']
            if processed_example is not None:
                test_data.append(processed_example)
                all_test_data.append(processed_example)
        dataset_name = test_dataset.value.lower()
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_dir, f'test_{test_datasets_name}.parquet')) #test parquet for each code dataset
    #save all code dataset
    all_test_df = pd.DataFrame(all_test_data)
    all_test_df.to_parquet(os.path.join(local_dir, 'test_code.parquet')) #test parquet for all code dataset
    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)