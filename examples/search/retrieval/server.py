#!/usr/bin/env python3
"""
Local retrieval server for Search-R1 training.
This server provides E5 embeddings + FAISS dense indexing and BM25 sparse retrieval.

Usage:
    python server.py --index_dir ./indices --port 8000
"""

import os
import json
import argparse
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

import faiss
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class LocalRetriever:
    """Local retrieval system with dense and sparse retrieval."""
    
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.corpus = []
        self.dense_index = None
        self.sparse_index = None
        self.encoder = None
        
        self._load_indices()
    
    def _load_indices(self):
        """Load prebuilt indices and corpus."""
        print(f"Loading indices from {self.index_dir}")
        
        # Load corpus
        corpus_file = self.index_dir / "corpus.json"
        if corpus_file.exists():
            with open(corpus_file, 'r') as f:
                self.corpus = json.load(f)
            print(f"Loaded corpus with {len(self.corpus)} documents")
        else:
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        # Load dense index
        dense_index_file = self.index_dir / "dense_index.faiss"
        if dense_index_file.exists():
            self.dense_index = faiss.read_index(str(dense_index_file))
            print(f"Loaded dense index with {self.dense_index.ntotal} vectors")
        else:
            print("Warning: Dense index not found")
        
        # Load sparse index
        sparse_index_file = self.index_dir / "sparse_index.json"
        if sparse_index_file.exists():
            with open(sparse_index_file, 'r') as f:
                sparse_data = json.load(f)
            
            # Reconstruct BM25 index
            tokenized_corpus = [doc.split() for doc in sparse_data['tokenized_corpus']]
            self.sparse_index = BM25Okapi(tokenized_corpus)
            print(f"Loaded sparse index with {len(tokenized_corpus)} documents")
        else:
            print("Warning: Sparse index not found")
        
        # Load encoder
        encoder_path = self.index_dir / "encoder"
        if encoder_path.exists():
            self.encoder = SentenceTransformer(str(encoder_path))
            print("Loaded sentence encoder")
        else:
            # Fallback to default encoder
            print("Loading default E5 encoder...")
            self.encoder = SentenceTransformer('intfloat/e5-base-v2')
    
    def dense_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform dense retrieval using FAISS."""
        if self.dense_index is None or self.encoder is None:
            return []
        
        # Encode query
        query_vector = self.encoder.encode([f"query: {query}"])
        query_vector = query_vector.astype('float32')
        
        # Search
        scores, indices = self.dense_index.search(query_vector, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.corpus):
                results.append({
                    "content": self.corpus[idx],
                    "score": float(score),
                    "type": "dense"
                })
        
        return results
    
    def sparse_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform sparse retrieval using BM25."""
        if self.sparse_index is None:
            return []
        
        # Tokenize query
        query_tokens = query.split()
        
        # Get BM25 scores
        scores = self.sparse_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.corpus) and scores[idx] > 0:
                results.append({
                    "content": self.corpus[idx],
                    "score": float(scores[idx]),
                    "type": "sparse"
                })
        
        return results
    
    def hybrid_search(self, query: str, k: int = 10, dense_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval combining dense and sparse results."""
        sparse_weight = 1.0 - dense_weight
        
        # Get results from both methods
        dense_results = self.dense_search(query, k * 2)  # Get more to rerank
        sparse_results = self.sparse_search(query, k * 2)
        
        # Normalize scores
        if dense_results:
            max_dense = max(r['score'] for r in dense_results)
            for r in dense_results:
                r['score'] = r['score'] / max_dense if max_dense > 0 else 0
        
        if sparse_results:
            max_sparse = max(r['score'] for r in sparse_results)
            for r in sparse_results:
                r['score'] = r['score'] / max_sparse if max_sparse > 0 else 0
        
        # Combine results
        combined_scores = {}
        
        for result in dense_results:
            content = result['content']
            combined_scores[content] = dense_weight * result['score']
        
        for result in sparse_results:
            content = result['content']
            if content in combined_scores:
                combined_scores[content] += sparse_weight * result['score']
            else:
                combined_scores[content] = sparse_weight * result['score']
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for content, score in sorted_results[:k]:
            results.append({
                "content": content,
                "score": score,
                "type": "hybrid"
            })
        
        return results
    
    def search(self, query: str, method: str = "hybrid", k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Main search interface."""
        if method == "dense":
            return self.dense_search(query, k)
        elif method == "sparse":
            return self.sparse_search(query, k)
        elif method == "hybrid":
            dense_weight = kwargs.get("dense_weight", 0.7)
            return self.hybrid_search(query, k, dense_weight)
        else:
            raise ValueError(f"Unknown search method: {method}")


# Flask app
app = Flask(__name__)
retriever = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "corpus_size": len(retriever.corpus) if retriever else 0,
        "dense_index_loaded": retriever.dense_index is not None if retriever else False,
        "sparse_index_loaded": retriever.sparse_index is not None if retriever else False
    })


@app.route('/retrieve', methods=['POST'])
def retrieve():
    """Main retrieval endpoint."""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request"}), 400
        
        query = data['query']
        method = data.get('method', 'hybrid')
        k = data.get('k', 10)
        dense_weight = data.get('dense_weight', 0.7)
        
        # Perform search
        results = retriever.search(
            query=query,
            method=method,
            k=k,
            dense_weight=dense_weight
        )
        
        return jsonify({
            "query": query,
            "method": method,
            "results": results,
            "num_results": len(results)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Local retrieval server")
    parser.add_argument("--index_dir", default="./indices", help="Directory containing indices")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Initialize retriever
    global retriever
    try:
        retriever = LocalRetriever(args.index_dir)
        print(f"Retrieval server initialized with {len(retriever.corpus)} documents")
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        return
    
    # Start server
    print(f"Starting retrieval server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main() 