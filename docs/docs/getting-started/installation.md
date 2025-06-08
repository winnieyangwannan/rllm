# Installation Guide

This guide will help you set up rLLM on your system.

## Prerequisites

Before installing rLLM, ensure you have the following:

- Python 3.8 or higher
- Git
- pip (Python package installer)
- CUDA-compatible GPU (recommended for training)

## Basic Installation

Follow these steps to install rLLM:

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/agentica-project/rllm-internal.git

# Navigate to the project directory
cd rllm

# Install dependencies
pip install -e ./verl[vllm,gpu,sglang]
pip install -e .
pip install -r requirements.txt
```

This will install rLLM and all its dependencies in development mode。

## Troubleshooting

If you encounter any issues during installation:

1. Make sure your Python version is compatible (3.8+)
2. Check that all dependencies were installed correctly
3. For GPU-related issues, ensure your CUDA drivers are up to date
4. Verify that all environment variables are set correctly

For more help, refer to the [GitHub issues page](https://github.com/agentica-project/rllm-internal/issues). 