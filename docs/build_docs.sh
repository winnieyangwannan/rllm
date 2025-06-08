#!/bin/bash

# Install mkdocs and required plugins if not already installed
pip install mkdocs mkdocs-material

# Build the documentation
cd "$(dirname "$0")"
mkdocs build

# Serve the documentation (if requested)
if [ "$1" == "serve" ]; then
    mkdocs serve
fi

echo "Documentation built successfully!"
echo "To serve the documentation, run: cd docs && mkdocs serve" 