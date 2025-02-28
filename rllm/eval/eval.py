import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import csv

from rllm.data.dataset_types import TrainDataset, TestDataset
from rllm.sampler import DistributedSampler
from rllm.eval.utils import evaluate_dataset

def parse_args(parser: argparse.ArgumentParser):
    """Parse command line arguments for trajectory generation."""
    parser.add_argument('--datasets', type=str, nargs='+', 
                       choices=['AIME', 'AMC', 'MATH', 'OMNI_MATH', 'OLYMPIAD', 'MINERVA', 'OLYMPIAD_BENCH'],
                       default=['AIME'], help='Datasets to process')
    parser.add_argument('--split', type=str, choices=['train', 'test'],
                       default='train', help='Dataset split to use')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of vLLM workers')
    parser.add_argument('--tensor_parallel_size', type=int, default=8,
                       help='Tensor parallelism per worker')
    parser.add_argument('--model', type=str, default="Qwen/QwQ-32B-Preview",
                       help='Model name/path to use')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Temperature for sampling')
    parser.add_argument('--n', type=int, default=8,
                       help='Number of samples to generate per problem')
    parser.add_argument('--max_tokens', type=int, default=32000,
                        help='Maximum number of output tokens')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: current directory)')
    parser.add_argument('--engine', type=str, default="vllm", help="serving engine")

    return parser.parse_args()

def evaluate_dataset_wrapper(args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Wrapper function for evaluate_dataset to work with ThreadPoolExecutor"""
    return evaluate_dataset(
        dataset=args['dataset'],
        output_dir=args['output_dir'],
        engine=args['engine'],
        n=args['n'],
        temperature=args['temperature'],
        max_tokens=args['max_tokens'],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate trajectories for math problems')
    args = parse_args(parser)
    
    # Initialize the distributed VLLM engine
    engine = DistributedSampler(
        backend=args.engine,
        num_workers=args.num_workers,
        tensor_parallel_size=args.tensor_parallel_size,
        model=args.model
    )
    
    # Set output directory
    if args.output_dir:
        output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory set to: {output_dir}")
    
    # Process each dataset in parallel
    dataset_enum = TrainDataset.Math if args.split == 'train' else TestDataset.Math
    results_summary = {}
    
    # Prepare arguments for each dataset
    eval_tasks = [
        {
            'dataset': dataset_enum[dataset_name],
            'output_dir': output_dir,
            'engine': engine,
            'n': args.n,
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
        }
        for dataset_name in args.datasets
    ]
    
    # Run evaluations in parallel
    with ThreadPoolExecutor(max_workers=len(args.datasets)) as executor:
        results = executor.map(evaluate_dataset_wrapper, eval_tasks)
        
    # Collect results
    for dataset_name, stats in results:
        results_summary[dataset_name] = stats
        
    # Print final summary and save to CSV
    print("\nEvaluation Summary:")
    
    # Prepare CSV output
    csv_file = os.path.join(output_dir, f"evaluation.csv")
    csv_headers = ['Dataset', 'Total Problems', 'Pass@1', f'Pass@{args.n}', 'Pass@1 Average']
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
        for dataset_name, stats in results_summary.items():
            # Console output
            print(f"\n{dataset_name}:")
            print(f"Total problems: {stats['total_problems']}")
            print(f"Pass@1: {stats['pass@1']:.2%}")
            print(f"Pass@{args.n}: {stats[f'pass@{args.n}']:.2%}")
            print(f"Pass@1 average: {stats['pass@1_average']:.2%}")
            
            # CSV output
            writer.writerow([
                dataset_name,
                stats['total_problems'],
                f"{stats['pass@1']:.2%}",
                f"{stats[f'pass@{args.n}']:.2%}",
                f"{stats['pass@1_average']:.2%}"
            ])
    
    print(f"\nResults saved to: {csv_file}")
