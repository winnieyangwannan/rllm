import os
import gdown
import shutil

# Define the Google Drive file IDs for the JSON files
FILE_IDS = {
    "test_taco.json": "1T3IKyJceSPijxbKsJyWkFwPKU0eTVC3I",
    "train_taco.json": "1fA7SzLjCKP2u33n_WkBytN2e6iDxKL6h",
    "olympiad.json": "1TxTUkXR5WIXS1586XbkrkfpKLEyHiNJS",
}

# Define the destination paths
DEST_PATHS = {
    "test_taco.json": "rllm/data/test/code/taco.json",
    "train_taco.json": "rllm/data/train/code/taco.json",
    "olympiad.json": "rllm/data/train/math/olympiad.json",
}

# Create the necessary directories
for path in DEST_PATHS.values():
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Download and move files
for filename, file_id in FILE_IDS.items():
    temp_path = f"./{filename}"  # Download location
    dest_path = DEST_PATHS[filename]

    # Download the file
    print(f"Downloading {filename}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", temp_path, quiet=False)

    # Move to the correct location
    print(f"Moving {filename} to {dest_path}...")
    shutil.move(temp_path, dest_path)

print("All files downloaded and moved successfully.")