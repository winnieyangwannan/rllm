# Installation Guide

This guide will help you set up rLLM on your system.

## Prerequisites

Before installing rLLM, ensure you have the following:

- Python 3.10 or higher
- CUDA version >= 12.1

## Basic Installation

rLLM uses [verl](https://github.com/volcengine/verl) as its training backend. Follow these steps to install rLLM and our custom fork of verl:

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/agentica-project/rllm.git

conda env create -n rllm python=3.10
conda activate rllm

# Install dependencies
cd rllm
pip install -e ./verl[vllm,gpu,sglang]
pip install -e .
pip install -r requirements.txt
```

This will install rLLM and all its dependencies in development mode。

For more help, refer to the [GitHub issues page](https://github.com/agentica-project/rllm/issues). 