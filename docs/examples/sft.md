# Math SFT Training Example

This example demonstrates supervised fine-tuning (SFT) of math reasoning models using the rLLM framework. The SFT training pipeline generates high-quality trajectories from a teacher model and fine-tunes a student model on the successful trajectories using Qwen/Qwen2.5-Math-1.5B as the base model and agentica-org/DeepScaleR-1.5B-Preview as the teacher model.

## Overview

The Math SFT examples demonstrate:

- How to generate high-quality training data from teacher model trajectories
- How to perform supervised fine-tuning on successful math reasoning trajectories
- How to fine-tune math reasoning models using the DeepScaleR dataset

## Quick Start

### Setup Math Data

First, prepare your SFT training datasets:

```bash
cd examples/sft
python generate_sft_data.py --num_samples 1000 --reward_threshold 1.0 --output large_sft_data.parquet
```

### Model Hosting

Start a model server for the teacher model:

**Using vLLM for Data Generation**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model agentica-org/DeepScaleR-1.5B-Preview \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

### Generate SFT Data

Generate training data from teacher model trajectories:

```bash
python generate_sft_data.py --num_samples 1000 --reward_threshold 1.0 --output large_sft_data.parquet
```

### Train Math SFT Model

Train your math reasoning model with supervised fine-tuning:

```bash
bash train_math_sft.sh
```

## Code Reference

### SFT Data Generator

Main script for generating SFT training data:

```python title="examples/sft/generate_sft_data.py"
--8<-- "examples/sft/generate_sft_data.py"
```

### Math SFT Evaluator

Script for evaluating SFT model performance:

```python title="examples/sft/eval_math_sft.py"
--8<-- "examples/sft/eval_math_sft.py"
```

For detailed setup instructions, see the [README](https://github.com/agentica-project/rllm/blob/main/examples/sft/README.md) in the sft example directory.
