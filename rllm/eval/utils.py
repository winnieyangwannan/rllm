import concurrent.futures
import json
import os

from tqdm import tqdm

from rllm.data.dataset_types import Dataset
from rllm.data.utils import load_dataset
from rllm.sampler import DistributedSampler
from rllm.system_prompts import DEEPSEEK_MATH_SYSTEM_PROMPT
from rllm.rewards import RewardInput, RewardType, RewardConfig
from rllm.rewards.math_reward import RewardMathFn


def evaluate_dataset_entry(idx, engine, entry, n=8, temperature=0.6):
    """
    Process a single problem using the distributed VLLM engine.
    
    Returns:
    - dict: Problem record with trajectory and grade
    """
    problem = entry['problem']
    answer = entry['answer']
    if isinstance(answer, float) or isinstance(answer, int):
        answer = str(answer)
    content_dict = [
        {"role": "user", "content":  problem + ' ' + DEEPSEEK_MATH_SYSTEM_PROMPT},
    ]

    # Use the distributed engine's chat_completion method
    retry_limit = 5
    for retry_idx in range(retry_limit):
        try:
            sample_batch = engine.chat_completion(content_dict,
                                              n=n,
                                              temperature=temperature,
                                              max_tokens=24000)
            # Extract responses from Sample objects in the batch
            llm_responses = [sample.response for sample in sample_batch.samples]
            break
        except Exception as e:
            print(f"Error getting completion: {e}")
            llm_responses = None
            if retry_idx == retry_limit - 1:
                raise e
    
    reward_fn = RewardMathFn(RewardConfig(use_math_orm=True))
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
    return idx, entry


def evaluate_dataset(dataset: Dataset, output_dir: str, engine: DistributedSampler, n=8, temperature=0.6):
    print(f"\nProcessing dataset: {dataset}")
    problems = load_dataset(dataset)
    
    # Set output file path
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use dataset enum value for filename
    output_file = os.path.join(output_dir, f"{dataset.value}.json")
    # Convert to absolute path if relative
    if not os.path.isabs(output_file):
        output_file = os.path.abspath(output_file)
    
    results = {}

    # Process problems in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = [
            executor.submit(
                evaluate_dataset_entry,
                idx,
                engine,
                entry,
                n=n,
                temperature=temperature
            ) for idx, entry in enumerate(problems)
        ]

        # Process results as they complete
        for _, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), 
                                    total=len(futures), 
                                    desc=f"Processing {dataset}"):
            try:
                idx, entry = future.result()
                results[idx] = entry
            except Exception as e:
                print(f"Error processing problem in {dataset}: {e}")

    # Save final results for this dataset
    with open(output_file, mode="w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Calculate statistics
    total_problems = len(results)
    pass_at_1 = sum(entry['pass@1'] for entry in results.values()) / float(total_problems)
    pass_at_1_average = sum([sum(entry['grades']) for entry in results.values()]) / float(total_problems * n)
    pass_at_n = sum(entry[f'pass@{n}'] for entry in results.values()) / float(total_problems)
    
    return dataset, {
        'total_problems': total_problems,
        'pass@1': pass_at_1,
        f'pass@{n}': pass_at_n,
        'pass@1_average': pass_at_1_average
    }