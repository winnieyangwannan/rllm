# %%
import json
import pandas as pd
from datasets import load_dataset

ds = load_dataset("MatrixStudio/Codeforces-Python-Submissions", split="train")

print(ds)

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
        "problem": entry["prompt"] + "\n" + entry["problem-description"],#Note(Xiao):this may have problem
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
        "test_cases": entry["examples"],#str, when use it as a reward, need use json.loads
    }
    dataset.append(new_entry)

print(f"test dataset size: {len(dataset)}")

with open("test_codeforces.json", "w") as f:
    json.dump(dataset, f, indent=4)

# %%



