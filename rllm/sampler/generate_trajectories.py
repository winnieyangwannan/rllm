from copy import deepcopy
import os
from tqdm import tqdm

from rllm.system_prompts import COT_MATH_SYSTEM_PROMPT
from rllm.data.load_dataset import TrainDataset, TestDataset, load_dataset
from rllm.rewards.math.sympy_checker import grade_answer

import requests
import json

from rllm.sampler.distributed_sampler import DistributedVLLM
import concurrent.futures
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate trajectories for math problems')
    parser.add_argument('--dataset', type=str, choices=['AIME', 'AMC', 'MATH', 'OMNI_MATH', 'OLYMPIAD'],
                       default='AIME', help='Dataset to process')
    parser.add_argument('--split', type=str, choices=['train', 'test'],
                       default='train', help='Dataset split to use')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of vLLM workers')
    parser.add_argument('--tensor_parallel_size', type=int, default=2,
                       help='Tensor parallelism per worker')
    parser.add_argument('--model', type=str, default="Qwen/QwQ-32B-Preview",
                       help='Model name/path to use')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling')
    parser.add_argument('--n', type=int, default=8,
                       help='Number of samples to generate per problem')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: {dataset}_{split}_trajectories.json)')
    return parser.parse_args()

def generate_trajectory(idx, engine, entry, n=8, temperature=1.0):
    """
    Process a single problem using the distributed VLLM engine.
    
    Returns:
    - dict: Problem record with trajectory and grade
    """
    problem = entry['problem']
    answer = entry['answer']
    if 'trajectories' in entry and len(entry['trajectories']) >= n:
        return idx, entry
    content_dict = [
        {"role": "system", "content": COT_MATH_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    # Use the distributed engine's chat_completion method
    for _ in range(5):
        try:
            response_data = engine.chat_completion(content_dict,
                                                n=n,
                                                temperature=temperature)
            # Multiple responses
            llm_responses = [r['message']['content'] for r in response_data['choices']]
            break
        except Exception as e:
            print(f"Error getting completion: {e}")
            llm_responses = None

    # Grade the answer
    grades = [grade_answer(r if r else "", str(answer)) for r in llm_responses]
    # Convert grades to 0 and 1s
    grades = [1 if g else 0 for g in grades]
    # Compute pass@1 and pass@8
    pass_at_1 = grades[0]
    pass_at_8 = 1 if any(grades) else 0
    entry.update({
        'grades': grades,
        'pass@1': pass_at_1, 
        'pass@8': pass_at_8,
        'trajectories': llm_responses
    })
    return idx, entry

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize the distributed VLLM engine, do not cntrl C here...
    engine = DistributedVLLM(
        num_workers=args.num_workers,
        tensor_parallel_size=args.tensor_parallel_size,
        model=args.model
    )
    print('Engine initialized. Ready to generate trajectories.')
    
    # Load problems based on args
    dataset_enum = TrainDataset if args.split == 'train' else TestDataset
    problems = load_dataset(dataset_enum[args.dataset])
    
    # Set output file path
    output_file = args.output or f"{args.dataset.lower()}_{args.split}_trajectories.json"
    if not os.path.isabs(output_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, output_file)
    results = deepcopy(problems)
    
    # Process problems in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = [
            executor.submit(
                generate_trajectory,
                idx, 
                engine, 
                entry,
                n=args.n,
                temperature=args.temperature
            ) for idx, entry in enumerate(problems)
        ]

        # Process results as they complete
        for counter, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc="Processing problems"):
            try:
                idx, entry = future.result()
                results[idx] = entry
                
                # Save incrementally
                if counter % 50 == 0:
                    with open(output_file, mode="w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)      
            except Exception as e:
                print(f"Error processing problem: {e}")
    
    with open(output_file, mode="w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("All problems processed and appended to JSON.")
