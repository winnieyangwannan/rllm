#!/usr/bin/env python3
"""
Download and setup data for Search-R1 style training.
This script downloads Wikipedia corpus and Natural Questions dataset.

Usage:
    python download_search_data.py --data_dir ./search_data
"""

import os
import json
import argparse
import gzip
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def download_wikipedia_corpus(save_path: str):
    """Download Wikipedia corpus (wiki-18.jsonl format) from HuggingFace using the Search-R1 approach."""
    print("Downloading Wikipedia corpus from PeterJinGo/wiki-18-corpus...")
    
    wiki_dir = os.path.join(save_path, "wikipedia")
    os.makedirs(wiki_dir, exist_ok=True)
    
    wiki_file = os.path.join(wiki_dir, "wiki-18.jsonl")
    wiki_gz_file = os.path.join(wiki_dir, "wiki-18.jsonl.gz")
    
    if os.path.exists(wiki_file):
        print(f"Wikipedia corpus already exists at {wiki_file}")
        return wiki_file
    
    try:
        # Download the compressed corpus file using the exact Search-R1 approach
        print("Downloading wiki-18.jsonl.gz from PeterJinGo/wiki-18-corpus...")
        
        hf_hub_download(
            repo_id="PeterJinGo/wiki-18-corpus",
            filename="wiki-18.jsonl.gz",
            repo_type="dataset",
            local_dir=wiki_dir,
        )
        
        print(f"Downloaded compressed corpus to {wiki_gz_file}")
        
        # Extract the compressed file
        print("Extracting wiki-18.jsonl.gz...")
        try:
            # Try UTF-8 first
            with gzip.open(wiki_gz_file, 'rt', encoding='utf-8') as f_in:
                with open(wiki_file, 'w', encoding='utf-8') as f_out:
                    # Simple extraction without counting (to avoid re-reading)
                    for line_num, line in enumerate(tqdm(f_in, desc="Extracting corpus")):
                        f_out.write(line)
                        if line_num % 100000 == 0:  # Progress update every 100k lines
                            tqdm.write(f"Processed {line_num:,} lines")
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying with binary mode...")
            # Fallback to binary mode
            with gzip.open(wiki_gz_file, 'rb') as f_in:
                with open(wiki_file, 'wb') as f_out:
                    # Copy in chunks
                    chunk_size = 8192
                    for chunk in tqdm(iter(lambda: f_in.read(chunk_size), b''), desc="Extracting corpus"):
                        f_out.write(chunk)
        
        print(f"Wikipedia corpus extracted to {wiki_file}")
        
        # Clean up compressed file to save space
        os.remove(wiki_gz_file)
        print("Removed compressed file to save space")
        
        return wiki_file
        
    except Exception as e:
        print(f"Error downloading Wikipedia corpus: {e}")
        print("Make sure you have internet connection and can access HuggingFace hub.")
        return None


def download_natural_questions(save_path: str, max_train: int = 5000, max_val: int = 500):
    """Download and process Natural Questions dataset with size limits."""
    print(f"Downloading Natural Questions dataset (max {max_train} train, {max_val} val)...")
    
    nq_dir = os.path.join(save_path, "nq")
    os.makedirs(nq_dir, exist_ok=True)
    
    train_file = os.path.join(nq_dir, "nq_train.json")
    val_file = os.path.join(nq_dir, "nq_validation.json")
    
    # Check if already downloaded
    if os.path.exists(train_file) and os.path.exists(val_file):
        print(f"Natural Questions data already exists at {nq_dir}")
        return nq_dir
    
    # Download NQ dataset from HuggingFace with streaming to save disk space
    try:
        print("Loading Natural Questions dataset in streaming mode...")
        
        # Use streaming mode to save disk space
        dataset = load_dataset("natural_questions", trust_remote_code=True, streaming=True)
        
        # Process training data
        print(f"Processing training data (max {max_train} examples)...")
        train_data = []
        for i, example in enumerate(dataset['train']):
            if i >= max_train:
                break
            train_data.append(example)
            if i % 1000 == 0:
                print(f"Processed {i} training examples...")
        
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        # Process validation data
        print(f"Processing validation data (max {max_val} examples)...")
        val_data = []
        for i, example in enumerate(dataset['validation']):
            if i >= max_val:
                break
            val_data.append(example)
            if i % 100 == 0:
                print(f"Processed {i} validation examples...")
        
        with open(val_file, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"Natural Questions data saved to {nq_dir}")
        print(f"Saved {len(train_data)} training and {len(val_data)} validation examples")
        return nq_dir
        
    except Exception as e:
        print(f"Error downloading Natural Questions: {e}")
        print("This might be due to disk space or connection issues.")
        print("You can still use the training script - it will load data automatically.")
        return None


def download_prebuilt_indices(save_path: str):
    """Download pre-built E5 indices from Search-R1 (optional, faster than building from scratch)."""
    print("Downloading pre-built E5 indices from PeterJinGo/wiki-18-e5-index...")
    
    indices_dir = os.path.join(save_path, "prebuilt_indices")
    os.makedirs(indices_dir, exist_ok=True)
    
    try:
        # Download the pre-built index parts using the exact Search-R1 approach
        repo_id = "PeterJinGo/wiki-18-e5-index"
        for file in ["part_aa", "part_ab"]:
            print(f"Downloading {file}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                repo_type="dataset",
                local_dir=indices_dir,
            )
        
        print(f"Pre-built indices downloaded to {indices_dir}")
        print("Note: You'll need to concatenate part_aa and part_ab to create the full index")
        print("Run: cat part_aa part_ab > e5_Flat.index")
        
        return indices_dir
        
    except Exception as e:
        print(f"Warning: Could not download pre-built indices: {e}")
        print("You can still build indices from scratch using build_index.py")
        return None


def setup_search_data(data_dir: str = "./search_data", download_prebuilt: bool = False):
    """Setup all data needed for Search-R1 training."""
    print(f"Setting up Search-R1 data in {data_dir}")
    
    # Create data directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Download Wikipedia corpus
    wiki_file = download_wikipedia_corpus(data_dir)
    
    # Download Natural Questions
    nq_dir = download_natural_questions(data_dir)
    
    # Optionally download pre-built indices
    prebuilt_dir = None
    if download_prebuilt:
        prebuilt_dir = download_prebuilt_indices(data_dir)
    
    # Create summary
    summary = {
        "wikipedia_corpus": wiki_file,
        "natural_questions": nq_dir,
        "prebuilt_indices": prebuilt_dir,
        "setup_complete": wiki_file is not None and nq_dir is not None
    }
    
    summary_file = os.path.join(data_dir, "data_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nData setup {'completed' if summary['setup_complete'] else 'partially completed'}!")
    print(f"Summary saved to {summary_file}")
    
    if summary['setup_complete']:
        print("\nNext steps:")
        if prebuilt_dir:
            print("1. (Optional) Use pre-built indices or build your own:")
            print("   cd examples/search && python retrieval/build_index.py")
        else:
            print("1. Build retrieval indices: cd examples/search && python retrieval/build_index.py")
        print("2. Launch retrieval server: cd examples/search && bash retrieval/launch_server.sh")
        print("3. Start training: cd examples/search && python train_search_agent.py")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Download Search-R1 training data")
    parser.add_argument("--data_dir", default="./search_data", help="Directory to save data")
    parser.add_argument("--download_prebuilt", action="store_true", help="Also download pre-built E5 indices")
    
    args = parser.parse_args()
    
    setup_search_data(args.data_dir, args.download_prebuilt)


if __name__ == "__main__":
    main()