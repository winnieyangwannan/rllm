import json
from datasets import load_dataset

ds = load_dataset("codeparrot/apps", split="train", trust_remote_code=True)

print(ds)

# APPS dataset has difficulties as strings: introductory, interview, competition
# Let introductory->2.5, interview->5.5, competition->8.5
def difficulty_to_int(difficulty):
    if difficulty == "introductory":
        return 2.5
    if difficulty == "interview":
        return 5.5
    if difficulty == "competition":
        return 8.5
    return 0

dataset = []
for entry in ds:
    new_entry = {
        "problem": entry["question"],
        "solutions": entry["solutions"],
        "starter_code": entry["starter_code"],
        "url": entry["url"],
        "input_output":entry["input_output"],#str, when use it to test, need use json.load()
        "difficulty": difficulty_to_int(entry["difficulty"]),
    }
    dataset.append(new_entry)

print(len(dataset))
print(dataset[0])

with open("train_apps.json", "w") as f:
    json.dump(dataset, f, indent=4)


ds = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)

print(ds)


dataset = []
for entry in ds:
    new_entry = {
        "problem": entry["question"],
        "solutions": entry["solutions"],
        "starter_code": entry["starter_code"],
        "url": entry["url"],
        "input_output": entry["input_output"],#str, when use it to test, need use json.load()
        "difficulty": difficulty_to_int(entry["difficulty"]),
    }
    dataset.append(new_entry)

print(len(dataset))
print(dataset[0])

with open("test_apps.json", "w") as f:
    json.dump(dataset, f, indent=4)