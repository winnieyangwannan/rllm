import json
from datasets import load_dataset

ds = load_dataset("deepmind/code_contests", split="train")

print(ds)


dataset = []
for entry in ds:
    difficulty = entry["difficulty"] # TODO: understand how the difficulty is calculated and convert the difficulty to a number from 1-10 or 0 for unknown        
    new_entry = {
        "problem": entry["description"],
        "input_file": entry["input_file"],
        "output_file": entry["output_file"],
        "solutions": entry["solutions"],
        "incorrect_solutions": entry["incorrect_solutions"],
        "private_tests": entry["private_tests"],
        "generated_tests": entry["generated_tests"],
        "time_limit": entry["time_limit"],
        "public_tests": entry["public_tests"],#str, when use it as a reward, need use json.loads
        "memory_limit_bytes": entry["memory_limit_bytes"],
        "difficulty": difficulty,
    }
    dataset.append(new_entry)

print(len(dataset))

with open("train_code_contests.json", "w") as f:
    json.dump(dataset, f, indent=4)


ds = load_dataset("deepmind/code_contests", split="test")


dataset = []
for entry in ds:
    difficulty = entry["difficulty"] # TODO: understand how the difficulty is calculated and convert the difficulty to a number from 1-10 or 0 for unknown
    new_entry = {
        "problem": entry["description"],
        "input_file": entry["input_file"],
        "output_file": entry["output_file"],
        "solutions": entry["solutions"],
        "incorrect_solutions": entry["incorrect_solutions"],
        "private_tests": entry["private_tests"],
        "generated_tests": entry["generated_tests"],
        "time_limit": entry["time_limit"],
        "public_tests": entry["public_tests"],#str, when use it as a reward, need use json.loads
        "memory_limit_bytes": entry["memory_limit_bytes"],
        "difficulty": difficulty,
    }
    dataset.append(new_entry)

print(len(dataset))

with open("test_code_contests.json", "w") as f:
    json.dump(dataset, f, indent=4)