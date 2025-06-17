#!/usr/bin/env python3
"""
Build retrieval indices for Search-R1 training.
This script builds both dense (E5+FAISS) and sparse (BM25) indices from Wikipedia corpus.

Usage:
    python build_index.py --corpus_file ./search_data/wikipedia/wiki-18.jsonl --output_dir ./indices
"""

import argparse
import json
import os
import time

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_corpus(corpus_file: str, max_docs: int = None) -> list[str]:
    """Load corpus from JSONL file."""
    print(f"Loading corpus from {corpus_file}")

    corpus = []

    if corpus_file.endswith(".jsonl"):
        # Helper function to extract text from data
        def extract_text(data):
            if "contents" in data:  # Wikipedia corpus uses 'contents'
                text = data["contents"]
            elif "text" in data:
                text = data["text"]
            elif "content" in data:
                text = data["content"]
            elif "passage" in data:
                text = data["passage"]
            else:
                # Try to find any text field
                text_fields = ["title", "body", "snippet"]
                text = ""
                for field in text_fields:
                    if field in data:
                        text += " " + str(data[field])
                text = text.strip()
            return text

        # Try UTF-8 first, fall back to binary mode if needed
        try:
            with open(corpus_file, encoding="utf-8") as f:
                for i, line in enumerate(tqdm(f, desc="Loading corpus")):
                    if max_docs and i >= max_docs:
                        break

                    data = json.loads(line.strip())
                    text = extract_text(data)

                    if text:
                        corpus.append(text)
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, falling back to binary mode with error handling...")
            with open(corpus_file, "rb") as f:
                for i, raw_line in enumerate(tqdm(f, desc="Loading corpus")):
                    if max_docs and i >= max_docs:
                        break

                    try:
                        # Try to decode as UTF-8, skip if it fails
                        line = raw_line.decode("utf-8")
                    except UnicodeDecodeError:
                        try:
                            # Try latin-1 as fallback
                            line = raw_line.decode("latin-1")
                        except UnicodeDecodeError:
                            # Skip problematic lines
                            continue

                    try:
                        data = json.loads(line.strip())
                        text = extract_text(data)

                        if text:
                            corpus.append(text)
                    except json.JSONDecodeError:
                        # Skip malformed JSON lines
                        continue

    elif corpus_file.endswith(".json"):
        try:
            with open(corpus_file, encoding="utf-8") as f:
                data = json.load(f)
        except UnicodeDecodeError:
            with open(corpus_file, "rb") as f:
                data = json.load(f)

        if isinstance(data, list):
            for item in data[:max_docs] if max_docs else data:
                if isinstance(item, str):
                    corpus.append(item)
                elif isinstance(item, dict):
                    text = item.get("text", item.get("content", ""))
                    if text:
                        corpus.append(text)

    print(f"Loaded {len(corpus)} documents")
    return corpus


def build_dense_index(corpus: list[str], output_dir: str, model_name: str = "intfloat/e5-base-v2"):
    """Build dense index using sentence transformers and FAISS."""
    print(f"Building dense index with {model_name}")

    # Load encoder with offline mode handling
    try:
        encoder = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        print("Trying with local_files_only=True...")
        try:
            encoder = SentenceTransformer(model_name, local_files_only=True)
        except Exception as e2:
            print(f"Failed to load {model_name} locally: {e2}")
            print("Falling back to a basic model...")
            # Try simpler alternatives that might be available
            fallback_models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
            encoder = None
            for fallback in fallback_models:
                try:
                    print(f"Trying {fallback}...")
                    encoder = SentenceTransformer(fallback, local_files_only=True)
                    model_name = fallback
                    break
                except Exception:
                    continue

            if encoder is None:
                raise Exception("No sentence transformer model available. Please ensure internet connectivity or pre-download a model.") from None

    # Encode corpus
    print("Encoding documents...")
    # Add "passage: " prefix for E5 models
    prefixed_corpus = [f"passage: {doc}" for doc in corpus]

    # Encode in batches to manage memory
    batch_size = 32
    embeddings = []

    for i in tqdm(range(0, len(prefixed_corpus), batch_size), desc="Encoding"):
        batch = prefixed_corpus[i : i + batch_size]
        batch_embeddings = encoder.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)

    # Concatenate all embeddings
    embeddings = np.vstack(embeddings).astype("float32")

    print(f"Created embeddings with shape: {embeddings.shape}")

    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]

    # Use IndexFlatIP for inner product (cosine similarity)
    index = faiss.IndexFlatIP(dimension)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Add to index
    index.add(embeddings)

    print(f"Built FAISS index with {index.ntotal} vectors")

    # Save index
    os.makedirs(output_dir, exist_ok=True)
    index_file = os.path.join(output_dir, "dense_index.faiss")
    faiss.write_index(index, index_file)
    print(f"Saved dense index to {index_file}")

    # Save encoder
    encoder_dir = os.path.join(output_dir, "encoder")
    encoder.save(encoder_dir)
    print(f"Saved encoder to {encoder_dir}")

    return index_file


def build_sparse_index(corpus: list[str], output_dir: str):
    """Build sparse BM25 index."""
    print("Building sparse BM25 index...")

    # Tokenize corpus
    print("Tokenizing documents...")
    tokenized_corpus = []
    for doc in tqdm(corpus, desc="Tokenizing"):
        # Simple whitespace tokenization (can be improved)
        tokens = doc.lower().split()
        tokenized_corpus.append(tokens)

    # Build BM25 index
    print("Building BM25 index...")
    BM25Okapi(tokenized_corpus)

    # Save tokenized corpus (needed to reconstruct BM25)
    os.makedirs(output_dir, exist_ok=True)
    sparse_file = os.path.join(output_dir, "sparse_index.json")

    sparse_data = {"tokenized_corpus": [" ".join(tokens) for tokens in tokenized_corpus], "num_docs": len(tokenized_corpus)}

    with open(sparse_file, "w") as f:
        json.dump(sparse_data, f)

    print(f"Saved sparse index to {sparse_file}")
    return sparse_file


def build_indices(corpus_file: str, output_dir: str, max_docs: int = None, model_name: str = "intfloat/e5-base-v2"):
    """Build both dense and sparse indices."""
    start_time = time.time()

    # Load corpus
    corpus = load_corpus(corpus_file, max_docs)

    if not corpus:
        print("No documents found in corpus!")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save corpus
    corpus_file_out = os.path.join(output_dir, "corpus.json")
    with open(corpus_file_out, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved corpus to {corpus_file_out}")

    # Build dense index (skip if offline)
    try:
        dense_index_file = build_dense_index(corpus, output_dir, model_name)
    except Exception as e:
        print(f"Dense index building failed: {e}")
        print("Continuing with sparse index only...")
        dense_index_file = None

    # Build sparse index
    sparse_index_file = build_sparse_index(corpus, output_dir)

    # Create summary
    summary = {"corpus_file": corpus_file, "output_dir": output_dir, "num_documents": len(corpus), "model_name": model_name, "dense_index": dense_index_file, "sparse_index": sparse_index_file, "corpus_saved": corpus_file_out, "build_time_seconds": time.time() - start_time}

    summary_file = os.path.join(output_dir, "index_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nIndex building completed in {summary['build_time_seconds']:.2f} seconds!")
    print(f"Summary saved to {summary_file}")

    print("\nNext steps:")
    print(f"1. Start retrieval server: python retrieval/server.py --index_dir {output_dir}")
    print("2. Start training: python train_search_agent.py")


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indices for Search-R1")
    parser.add_argument("--corpus_file", required=True, help="Path to corpus file (JSONL or JSON)")
    parser.add_argument("--output_dir", default="./indices", help="Output directory for indices")
    parser.add_argument("--max_docs", type=int, help="Maximum number of documents to index")
    parser.add_argument("--model_name", default="intfloat/e5-base-v2", help="Sentence transformer model name")

    args = parser.parse_args()

    if not os.path.exists(args.corpus_file):
        print(f"Corpus file not found: {args.corpus_file}")
        print("Please run 'python data/download_search_data.py' first")
        return

    build_indices(corpus_file=args.corpus_file, output_dir=args.output_dir, max_docs=args.max_docs, model_name=args.model_name)


if __name__ == "__main__":
    main()
