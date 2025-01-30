import argparse
import os
import datasets
import pandas as pd
from rllm.data.dataloader import DatasetMix
from rllm.data.dataset_types import TrainDataset, TestDataset
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/deepscaler')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_datasets = [
        TrainDataset.DEEPSCALER
    ]
    train_dataset = DatasetMix(train_datasets)
    test_datasets = [ TestDataset.AIME ]
    test_dataset = DatasetMix(test_datasets) 
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop('problem')
            data_source = example.pop('dataset')
            # if data_source == TrainDataset.MATH:                
            #     difficulty = example.pop('difficulty')
            #     if difficulty < 4.0:
            #         return None
            question = question + ' ' + instruction_following
            answer = example.pop('answer')

            data = {
                "data_source": "",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        return process_fn
    train_data = []
    process_fn = make_map_fn('train')
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)
    test_data = []
    process_fn = make_map_fn('test')
    for idx, example in enumerate(test_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            test_data.append(processed_example)
    print("train data size:", len(train_data))
    print("test data size:", len(test_data))
    # Convert to DataFrame and save as parquet
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_df = pd.DataFrame(test_data)
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)