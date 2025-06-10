#!/bin/bash
#
# Launch script for the local retrieval server.
#
# Usage:
#     bash launch_server.sh [index_dir] [port]
#

# Default values
INDEX_DIR=${1:-"./indices"}
PORT=${2:-8000}

echo "Starting local retrieval server..."
echo "Index directory: $INDEX_DIR"
echo "Port: $PORT"

# Check if indices exist
if [ ! -d "$INDEX_DIR" ]; then
    echo "Error: Index directory '$INDEX_DIR' not found!"
    echo "Please build indices first:"
    echo "  python examples/search/retrieval/build_index.py --corpus_file ./search_data/wikipedia/wiki-18.jsonl"
    exit 1
fi

# Check for required files
if [ ! -f "$INDEX_DIR/corpus.json" ]; then
    echo "Error: corpus.json not found in $INDEX_DIR"
    echo "Please rebuild indices with build_index.py"
    exit 1
fi

# Start server
echo "Launching server..."
python examples/search/retrieval/server.py --index_dir "$INDEX_DIR" --port "$PORT" --host 0.0.0.0

echo "Server stopped." 