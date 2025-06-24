# DeepCoder Training Examples

This directory contains examples for training and running DeepCoder, a code reasoning LLM fine-tuned from DeepSeek-R1-Distill-Qwen-14B using distributed reinforcement learning (RL) to scale up to long context lengths. The model achieves 60.6% Pass@1 accuracy on LiveCodeBench v5, representing an 8% improvement over the base model and achieving similar performance to OpenAI's o3-mini.

Our examples uses the following:
* DeepSeek-R1-Distill-Qwen-14B as the base model
* LiveCodeBench v5 dataset for training
* LiveCodeBench v2 dataset for evaluation


## Model Hosting

### Option 1: Using vLLM

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 \
    --max-model-len 32768
```

### Option 2: Using SGLang

```bash
python -m sglang_router.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \ 
    --dp-size 1 \
    --dtype bfloat16
# increase dp_size to enable data-parallel processing on multi-GPU 
```

The server should be accessible at `http://localhost:30000/v1`

## Dataset Preparation

Prepare the required datasets (LiveCodeBench v2 for testing, LiveCodeBench v5 for training):

```bash
cd examples/deepcoder
python prepare_deepcoder_data.py
```

This will:
- Download LiveCodeBench v5 for training data (older problems from May 2023-July 2024)
- Download LiveCodeBench v2 for test data (recent problems from August 2024-January 2025)
- Register both datasets with the RLLM DatasetRegistry

## Running Inference

Once your model server is running and datasets are prepared, you can run inference:

```bash
cd examples/deepcoder
python evaluate_deepcoder.py
```

### Configuration Options

You can modify the inference script parameters:

- `n_parallel_agents`: Number of parallel agents (default: 32)
- `model_name`: Model to use (default: "agentica-org/DeepCoder-14B-Preview")
- `base_url`: API server URL (default: "http://localhost:30000/v1")
- `max_response_length`: Maximum response length (default: 64000)
- `max_prompt_length`: Maximum prompt length (default: 24576)
- `temperature`: Sampling temperature (default: 0.6)
- `top_p`: Top-p sampling (default: 0.95)

The script will:
1. Load the LiveCodeBench test dataset
2. Repeat each problem 16 times for Pass@K evaluation
3. Run parallel and async trajectory collection using the agent execution engine
4. Evaluate results and report Pass@1 and Pass@K accuracy

## Training

### Basic Training

To train DeepCoder with iterative context lengthening (16K -> 32K -> 64K):

```bash
cd examples/deepcoder
bash train_deepcoder_16k.sh

# modify MODEL_PATH to the 16k checkpoint path before running the script.
bash train_deepcoder_32k.sh

# modify MODEL_PATH to the 32k checkpoint path before running the script
bash train_deepcoder_64k.sh
```
