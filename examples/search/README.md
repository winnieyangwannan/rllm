# Search Training with RLLM

## Dependencies

```bash
pip install faiss-gpu sentence-transformers rank-bm25 flask
```

## Local Retrieval Setup

### 1. Download Data

```bash

# From the RLLM root directory
python examples/search/data/download_search_data.py --data_dir ./search_data

# OR, change to search directory first
cd examples/search
python data/download_search_data.py --data_dir ./search_data
```

This downloads:
- Wikipedia corpus (wiki-18.jsonl format) from [PeterJinGo/wiki-18-corpus](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus)
- Natural Questions dataset
- Creates data summary

### 3. Build Retrieval Indices

```bash
# From the RLLM root directory
python examples/search/retrieval/build_index.py \
    --corpus_file ./search_data/wikipedia/wiki-18.jsonl \
    --output_dir ./indices \
    --max_docs 100000  # Limit for faster testing

# OR, from examples/search directory
cd examples/search
python retrieval/build_index.py \
    --corpus_file ./search_data/wikipedia/wiki-18.jsonl \
    --output_dir ./indices \
    --max_docs 100000
```

This creates:
- Dense index: E5 embeddings + FAISS
- Sparse index: BM25 with tokenized corpus
- Corpus file for the server

### 4. Launch Retrieval Server

```bash
# From the RLLM root directory
bash examples/search/retrieval/launch_server.sh ./indices 8500

# OR, from examples/search directory
cd examples/search
bash retrieval/launch_server.sh ./indices 8500
```

The server provides REST API endpoints:
- `GET /health` - Health check
- `POST /retrieve` - Search endpoint

### 5. Run Training with Local Retrieval

```bash
# From the RLLM root directory
export RETRIEVAL_SERVER_URL="http://127.0.0.1:8000"
python examples/search/train_search_agent.py

# OR, from examples/search directory
cd examples/search
export RETRIEVAL_SERVER_URL="http://127.0.0.1:8000"
python train_search_agent.py
```

## Training Script Details

The training script (`train_search_agent.py`) uses RLLM's existing infrastructure:

### Dataset Loading
```python
# Uses existing RLLM pattern from examples/search/run_search_agent.py
train_data, val_data = load_search_data(train_size=3000, test_size=100)
```
- Automatically loads HotpotQA + Natural Questions from HuggingFace
- Processes into RLLM format with proper prompts and metadata
- Registers with DatasetRegistry for reuse

### Search Configuration
Control search behavior via environment variables:

```bash
export RETRIEVAL_SERVER_URL="http://127.0.0.1:8000"  # Local server
export MAX_SEARCH_RESULTS=10                         # Results per query
```

### Training Configuration
Use standard RLLM training configuration:

```bash
python train_search_agent.py
```

## API Reference

#### Search
```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "method": "hybrid",
    "k": 5,
    "dense_weight": 0.7
  }'
```

Response:
```json
{
  "query": "What is the capital of France?",
  "method": "hybrid",
  "results": [
    {
      "content": "Paris is the capital and most populous city of France...",
      "score": 0.95,
      "type": "hybrid"
    }
  ],
  "num_results": 5
}
```

### Search Methods
- `dense`: E5 embeddings + FAISS similarity search
- `sparse`: BM25 keyword matching
- `hybrid`: Weighted combination (default: 70% dense, 30% sparse)

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
```bash
pip install faiss-gpu sentence-transformers rank-bm25 flask
```

2. **CUDA Out of Memory**
```bash
# Reduce batch size in build_index.py
python retrieval/build_index.py --max_docs 10000
```

3. **Server Won't Start**
```bash
# Check if indices were built
ls -la ./indices/
# Should contain: corpus.json, dense_index.faiss, sparse_index.json
```

### Performance Tips

1. **For Large Corpora**: Use `--max_docs` to limit corpus size during development
2. **Memory Optimization**: Build indices on a machine with sufficient RAM (16GB+ recommended)
3. **Speed vs Accuracy**: Dense search is more accurate, sparse is faster
4. **Hybrid Tuning**: Adjust `dense_weight` parameter (0.7 works well for most cases)
