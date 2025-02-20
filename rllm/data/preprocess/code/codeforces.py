# %%
import json
import pandas as pd
from datasets import load_dataset

ds = load_dataset("DenCT/codeforces-problems-7k", split="train")

print(ds)

# %%
ds[0]


# %%
TEMPLATE = """
{description}

Input Format:
{input_format}

Output Format:
{output_format}

Tags:
{tags}

Time Limit: {time_limit} ms
Memory Limit: {memory_limit} MB
"""

TEMPLATE_NO_LIMIT = """
{description}

Input Format:
{input_format}

Output Format:
{output_format}

Tags:
{tags}

Demo input: {demo_input}
Demo output: {demo_output}
"""


# %%
# func. to extract difficulty rating from tags
def extract_difficulty(tags):
    if pd.isnull(tags):  # Handles when tags are null
        return 0
    for tag in tags.split(","):
        if "*" in tag:  # Difficulty is marked with a '*' symbol
            try:
                return int(tag.replace("*", ""))
            except ValueError:
                continue
    return 0  # Default difficulty 

# %%
dataset = []
for entry in ds:
    new_entry = {
        "problem": TEMPLATE_NO_LIMIT.format(
            description=entry["problem-description"],
            input_format=entry["input-specification"],
            output_format=entry["output-specification"],
            tags=entry["tags"],
            demo_input=entry["demo-input"],
            demo_output=entry["demo-output"],
        ),
        "test_cases": entry["test_cases"],#str, when use it as a reward, need use json.loads
    }
    dataset.append(new_entry)

print(f"train dataset size: {len(dataset)}")

with open("train_codeforces.json", "w") as f:
    json.dump(dataset, f, indent=4)

# %%
ds = load_dataset("Qwen/CodeElo", split="test")

print(ds)

# %%
def process_test_cases(raw_cases):
    # The first element is the full input string, the second is the full output string
    formatted_cases = []
    for rc in raw_cases:
        input_case = rc[0]  # Keep input as a single string
        output_case = rc[1]  # Keep output as a single string

        # Structure the test cases
        formatted_cases.append({"input": input_case, "output": output_case})

    return formatted_cases

# %%
dataset = []
for entry in ds:
    new_entry = {
        "problem": TEMPLATE.format(
            description=entry["description"],
            input_format=entry["input"],
            output_format=entry["output"],
            tags=entry["tags"],
            time_limit=entry["time_limit_ms"],
            memory_limit=entry["memory_limit_mb"],
        ),
        "test_cases": process_test_cases(entry["examples"]),#str, when use it as a reward, need use json.loads
    }
    dataset.append(new_entry)

print(f"test dataset size: {len(dataset)}")

with open("test_codeforces.json", "w") as f:
    json.dump(dataset, f, indent=4)

# %%



