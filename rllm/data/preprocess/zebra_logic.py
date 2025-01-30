from datasets import load_dataset
import json
import math
import numpy as np

# Load the ZebraLogic dataset

dataset = load_dataset("allenai/ZebraLogicBench-private", "grid_mode")

# Function to convert a single example to JSON format
def convert_to_json(example):
    n, m = int(example["size"][0]), int(example["size"][-1])
    difficulty = calculate_difficulty(n, m)
    return {
        "id": example["id"],
        "size": example["size"],
        "puzzle": example["puzzle"],
        "difficulty": -(np.round(np.log10(difficulty), decimals=2)), # Can replace with calculate_difficulty(difficulty) for more discreteness
        "solution": example["solution"]
    }


def calculate_difficulty(n, m):
    return 1 / (math.factorial(n) ** m)

# https://github.com/WildEval/ZeroEval/blob/4149c25325ac3d45fa0c192e446b911e8ac3d524/data_prep/zebra_difficulty.py#L26 
def classify_difficulty(difficulty):
    difficulty = np.log10(difficulty)
    if difficulty > -4:
        return "easy"
    elif -6 < difficulty <= -4:
        return "medium"
    else:
        return "hard"

# Convert the entire dataset to JSON
json_data = [convert_to_json(example) for example in dataset["test"]]

# Calculate split sizes
total_samples = len(json_data)
train_size = int(0.8 * total_samples)

# Shuffle the data
np.random.seed(42)  # for reproducibility
np.random.shuffle(json_data)

# Split the data
train_data = json_data[:train_size]
test_data = json_data[train_size:]

# Save the train JSON data
with open("zebra_logic_train.json", "w") as json_file:
    json.dump(train_data, json_file, indent=4)

# Save the test JSON data
with open("zebra_logic_test.json", "w") as json_file:
    json.dump(test_data, json_file, indent=4)

print(f"Dataset split and saved:")
print(f"Train set: {len(train_data)} samples saved as zebra_logic_train.json")
print(f"Test set: {len(test_data)} samples saved as zebra_logic_test.json")
