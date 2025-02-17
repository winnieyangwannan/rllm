import json
from datasets import load_dataset

ds = load_dataset("BAAI/TACO", split="train", trust_remote_code=True)

print(ds)


# TACO dataset has difficulties as strings: EASY, MEDIUM, MEDIUM_HARD, HARD, VERY_HARD
# Let EASY->1.9, MEDIUM->3.7, MEDIUM_HARD->5.5, HARD->7.3, VERY_HARD->9.1
def difficulty_to_int(difficulty):
    if difficulty == "EASY":
        return 1.9
    elif difficulty == "MEDIUM":
        return 3.7
    elif difficulty == "MEDIUM_HARD":
        return 5.5
    elif difficulty == "HARD":
        return 7.3
    elif difficulty == "VERY_HARD":
        return 9.1
    else:
        return 0


dataset = []
for entry in ds:        
    new_entry = {
        "problem": entry["question"],
        "solutions": entry["solutions"],
        "starter_code": entry["starter_code"],
        "difficulty": entry["difficulty"],
        "Expected Time Complexity": entry["Expected Time Complexity"],
        "Expected Auxiliary Space": entry["Expected Auxiliary Space"],
        "input_output": entry["input_output"],# str, #str, when use it to test the model response, need use json.load()
        "time_limit": entry["time_limit"],
        "memory_limit": entry["memory_limit"],
        "url": entry["url"],
        "difficulty": difficulty_to_int(entry["difficulty"]),
    }
    dataset.append(new_entry)

print(len(dataset))
print(dataset[0])

with open("train_taco.json", "w") as f:
    json.dump(dataset, f, indent=4)



ds = load_dataset("BAAI/TACO", split="test", trust_remote_code=True)

print(ds)


dataset = []
for entry in ds:
    new_entry = {
        "problem": entry["question"],
        "solutions": entry["solutions"],
        "starter_code": entry["starter_code"],
        "difficulty": entry["difficulty"],
        "Expected Time Complexity": entry["Expected Time Complexity"],
        "Expected Auxiliary Space": entry["Expected Auxiliary Space"],
        "input_output": entry["input_output"], #str, when use it to test the model response, need use json.load()
        "time_limit": entry["time_limit"],
        "memory_limit": entry["memory_limit"],
        "url": entry["url"],
        "difficulty": difficulty_to_int(entry["difficulty"]),
    }
    dataset.append(new_entry)

print(len(dataset))
print(dataset[0])

with open("test_taco.json", "w") as f:
    json.dump(dataset, f, indent=4)

    