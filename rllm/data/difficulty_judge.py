"""
Labels approximate difficulty between 1-10 for each problem in the train/ or test/ dataset.
"""
import json
import concurrent.futures
from concurrent.futures import as_completed
from copy import deepcopy
from tqdm import tqdm

from rllm.system_prompts import MATH_DIFFICULTY_PROMPT
from rllm.utils import call_gemini_llm

def get_difficulty(idx, entry):
    """
    1) Extract problem and solution text.
    2) Call LLM for difficulty estimates (8 numeric strings).
    3) Convert to float safely, filter out parse errors.
    4) Take the average and store as 'difficulty'.
    """
    if entry.get('difficulty') is not None:
        # Skip if already computed
        return idx, entry
    problem_text = entry.get('problem', '')
    solution_text = entry.get('solution', '')

    # Pass@8 Difficulty.
    output_list = call_gemini_llm(
        f"Problem: {problem_text}\n----\nSolution: {solution_text}",
        system_prompt=MATH_DIFFICULTY_PROMPT,
        n=8,
        temperature=1.0,
    )

    # Filter out error messages
    # (Use .lower() to catch both uppercase/lowercase errors)
    output_list = [
        o for o in output_list
        if 'error' not in o.lower() and 'solution not found' not in o.lower()
    ]

    # Attempt to parse each string as float
    values = []
    for o in output_list:
        try:
            val = float(o)
            values.append(val)
        except ValueError:
            # Ignore anything that can't be parsed as float
            pass

    # Compute the average or set None if no valid floats
    if values:
        difficulty = sum(values) / len(values)
    else:
        difficulty = None
        print('Failed parsing all difficulties: ', output_list)

    # Add the difficulty field to the entry
    entry['difficulty'] = difficulty

    return idx, entry

if __name__ == "__main__":
    base_file = "train/olympiad.json"
    
    with open(base_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = deepcopy(data)
    # Use ThreadPoolExecutor to process concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        # Submit jobs to the executor, passing (index, entry)
        futures = [
            executor.submit(get_difficulty, i, entry)
            for i, entry in enumerate(data)
        ]

        # We'll track how many have completed
        done_count = 0
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, result = future.result()
            results[idx] = result
            done_count += 1

            # Periodically print progress and save partial results
            if done_count % 5000 == 0:
                print(f"Processed {done_count} entries.")
                with open(base_file, "w", encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    # Save final results
    with open(base_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
