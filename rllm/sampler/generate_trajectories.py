import argparse
from pathlib import Path
import concurrent.futures
import json
import os
from copy import deepcopy

from tqdm import tqdm

from rllm.data.dataset_types import TrainDataset, TestDataset
from rllm.data.utils import load_dataset
from rllm.sampler import DistributedSampler
from rllm.system_prompts import COT_MATH_SYSTEM_PROMPT
from rllm.rewards import RewardInput, RewardType, RewardConfig
from rllm.rewards.math_reward import RewardMathFn


def parse_args(parser: argparse.ArgumentParser):
    """Parse command line arguments for trajectory generation.
    
    Returns:
        argparse.Namespace: Parsed command line arguments with the following fields:
            - dataset (str): Dataset to process (AIME, AMC, MATH, OMNI_MATH, or OLYMPIAD)
            - split (str): Dataset split to use (train or test)
            - num_workers (int): Number of vLLM workers to use for parallel processing
            - tensor_parallel_size (int): Tensor parallelism size per worker for model sharding
            - model (str): Name or path of the model to use
            - temperature (float): Sampling temperature for generation
            - n (int): Number of samples to generate per math problem
            - output (str): Output file path for saving trajectories
    """
    parser.add_argument('--dataset', type=str, choices=['AIME', 'AMC', 'MATH', 'OMNI_MATH', 'OLYMPIAD'],
                       default='AIME', help='Dataset to process')
    parser.add_argument('--split', type=str, choices=['train', 'test'],
                       default='train', help='Dataset split to use')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of vLLM workers')
    parser.add_argument('--tensor_parallel_size', type=int, default=8,
                       help='Tensor parallelism per worker')
    parser.add_argument('--model', type=str, default="Qwen/QwQ-32B-Preview",
                       help='Model name/path to use')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Temperature for sampling')
    parser.add_argument('--n', type=int, default=8,
                       help='Number of samples to generate per problem')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: {dataset}_{split}_trajectories.json)')
    return parser.parse_args()

def generate_trajectory(idx, engine, entry, n=8, temperature=0.6):
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
    retry_limit = 5
    for retry_idx in range(retry_limit):
        try:
            sample_batch = engine.chat_completion(content_dict,
                                              n=n,
                                              temperature=temperature)
            # Extract responses from Sample objects in the batch
            llm_responses = [sample.response for sample in sample_batch.samples]
            print (len(llm_responses))
            break
        except Exception as e:
            print(f"Error getting completion: {e}")
            llm_responses = None
            if retry_idx == retry_limit - 1:
                raise e
    
    reward_fn = RewardMathFn(RewardConfig)
    reward_inputs = [RewardInput(problem=problem, problem_type=RewardType.MATH, model_response=r, metadata={"answer": answer}) for r in llm_responses]
    reward_outputs = [reward_fn(r) for r in reward_inputs]
    # Grade the answer
    grades = [1 if r.is_correct else 0 for r in reward_outputs]
    # Compute pass@1 and pass@8
    pass_at_1 = grades[0]
    pass_at_n = 1 if any(grades) else 0
    entry.update({
        'grades': grades,
        'pass@1': pass_at_1, 
        f'pass@{n}': pass_at_n,
        'trajectories': llm_responses
    })
    print(entry)
    return idx, entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate trajectories for math problems')
    args = parse_args(parser)
    
    # Initialize the distributed VLLM engine, do not cntrl C here...
    engine = DistributedSampler(
        backend="sglang",
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

    # Easy debug run just one: 
    quick_sequential_debug = False
    if quick_sequential_debug:
        for i in range (5):
            idx, entry = i, problems[i]
            generate_trajectory (idx, engine, entry, n=args.n, temperature=args.temperature)
        exit()
        
    problems = problems [:2]
        
    # check if output_file exists
    if not Path(output_file).exists():
        Path(output_file).touch()
        
    if os.stat(output_file).st_size==0:
        results = {}
    else:
        # output file has content
        results = json.load(open(output_file, "r"))
        if 'problem_number' not in problems[0].keys(): 
            raise NotImplementedError
        problems = [p for p in problems if p['problem_number'] not in results.keys()]
        
        

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
                # TODO: This really should be done with a file lock and done by appending lines
                if counter % 50 == 0:
                    with open(output_file, mode="w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)      
            except Exception as e:
                print(f"Error processing problem: {e}")
    with open(output_file, mode="w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("All problems processed and appended to JSON.")
