<h1 align="center"> DeepSWE-Preview - Training a State-of-the-Art Coding Agent by Scaling RL </h1>

<!-- paper . data and models . project page -->
<p align="center">
<a href="#">📃 Blog Post</a>
•
<a href="https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset" > 🤗 HF Dataset (R2E-Gym) </a>
•
<!-- project page -->
<a href="https://wandb.ai/mluo/deepswe" >🔥 WandB Logs</a>
•
<a href="https://huggingface.co/agentica-org/DeepSWE-Preview" > 🤗 DeepSWE-Preview</a>
•
<a href="https://drive.google.com/file/d/10LIwpJeaFuiX6Y-qEG2a4a335PEuQJeS/view?usp=sharing" > 📈 Evaluation Logs</a>
•
<a href="https://agentica-project.com/" > 🌐 Project Page</a>
•
<a href="https://github.com/agentica-project/rllm" > 🧑‍💻 Code</a>
</p>

<div align="center">

[![Github](https://img.shields.io/badge/RLLM-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/agentica-project/rllm)
[![Website](https://img.shields.io/badge/Site-%23000000.svg?style=for-the-badge&logo=semanticweb&logoColor=white)](https://www.agentica-project.com) 
[![Twitter](https://img.shields.io/badge/Agentica-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/Agentica_)
[![Hugging Face Collection](https://img.shields.io/badge/Agentica-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/agentica-org)

</div>

We introduce DeepSWE-Preview, a reasoning-enabled coding agent trained from scratch from Qwen3-32B with only reinforcement learning (RL). It achieves 59.2% on SWE-Bench-Verified with test-time scaling, reaching SOTA for open-weight coding agents (42.2% Pass@1, 71.0% Pass@16).

DeepSWE is trained using [**rLLM**](https://github.com/agentica-project/rllm), our framework for post-training language agents using high-quality SWE environments from [**R2E-Gym**](https://github.com/R2E-Gym/R2E-Gym). We’ve open-sourced everything—our dataset, code, training, and evaluation logs, for everyone to progress on scaling and improving agents with RL.

## Quick Start 🎯

### 1. 📦 Installation
```bash
# Installing Python 3.10 Environment.
conda create -n rllm python=3.10 -y
conda activate rllm

# Installing RLLM dependencies.
cd rllm
pip install -e ./verl
pip install -e .
```

Also, install [**R2E-Gym**](https://github.com/R2E-Gym/R2E-Gym) for high-quality SWE-Bench environments used for RL training.
```bash
git clone https://github.com/agentica-project/R2E-Gym.git
cd R2E-Gym
pip install -e .
```


### 2. 🤗 Data and Agent Scaffold

We use the R2E-Gym environments for RL training. R2E-Gym environment can be simply used as:
```python
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent
from pathlib import Path
from datasets import load_dataset

# load gym dataset
ds = load_dataset("R2E-Gym/R2E-Gym-Subset")
split = 'train'

# load gym environment
env_index = 100 # index of the environment [0, len(ds)]
env_args = EnvArgs(ds = ds[split][env_index])
env = RepoEnv(env_args)

# load agent
agent_args = AgentArgs.from_yaml(Path('./src/r2egym/agenthub/config/r2egym/edit_non_fn_calling.yaml'))
# define llm: ['claude-3-5-sonnet-20241022', 'gpt-4o', 'vllm/agentica-org/DeepSWE-Preview']
agent_args.llm_name = 'vllm/agentica-org/DeepSWE-Preview'
agent = Agent(name="EditingAgent", args=agent_args)

# run the agent
output = agent.run(env, max_steps=40)
```

## 🤖 3. Running DeepSWE-Preview Inference

First, start the VLLM server to serve the DeepCoder model:

```bash
# Start VLLM server with tensor parallelism across 8 GPUs
export MAX_CONTEXT_LEN=65536
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve agentica-org/DeepSWE-Preview \
    --tensor-parallel-size 8 \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching
```

> ⚠️ **Important**: Wait for the server to fully load before proceeding to the next step. You should see logs indicating the server is ready to accept requests.


In a new terminal session, run the DeepSWE agent evaluation:

```bash
# Activate the virtual environment (if in new terminal)
source .venv/bin/activate

# Run the DeepSWE agent on SWE-Bench Verified
time python src/r2egym/agenthub/run/edit.py runagent_multiple \
    --traj_dir "./traj" \
    --max_workers 48 \
    --start_idx 0 \
    --k 500 \
    --dataset "R2E-Gym/SWE-Bench-Verified" \
    --split "test" \
    --llm_name "openai/agentica-org/DeepSWE-Preview" \
    --scaffold "r2egym" \
    --use_fn_calling False \
    --exp_name "$EXP_NAME" \
    --temperature "$TEMP" \
    --max_steps_absolute 100 \
    --backend "docker" \
    --condense_history False \
    --max_reward_calc_time 1200 \
    --max_tokens 65536
```

**Parameter Explanation:**
- `--max_workers 54`: Number of parallel workers for processing, reduce if you hit trajectory time limit errors
- `--k 500`: Number of instances to evaluate (max 500 for SWE-Bench Verified)
- `--temperature 1`: Sampling temperature for model responses
- `--max_steps 40`: Maximum steps per trajectory
- `--max_steps_absolute 100`: Absolute maximum steps limit

> 📊 **Expected Runtime**: This evaluation may take several hours depending on your hardware configuration.

**Trajectory Visualization:** 
The generated trajectories are saved in `./traj` directory. You can visualize the trajectories using the trajectory visualization tool in [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym),
```bash
python app/app.py --traj_dir "./traj"
```

## 🔥 4. Training DeepSWE-Preview with rLLM and R2E-Gym

[TODO]

# 🔬 5. DeepSWE-Preview Reproduction Guide

Please refer the following for detailed reproduction guide for DeepSWE-Preview.
* [DeepSWE-Preview Reproduction Guide](https://github.com/agentica-project/R2E-Gym/edit/master/reproduction/DEEPSWE_REPRODUCTION.MD)
* [DeepSWE-Preview with Hybrid Test-time Scaling](https://github.com/agentica-project/R2E-Gym/blob/master/reproduction/DEEPSWE_TTS_REPRODUCTION.MD)

## Citation

```
@misc{deepswe2025,
  title={DeepSWE: Training a State-of-the-Art Coding Agent from Scratch by Scaling RL},
  author={Michael Luo and Naman Jain and Jaskirat Singh and Sijun Tan and Ameen Patel and Qingyang Wu and Alpay Ariyak and Colin Cai and Tarun Venkat and Shang Zhu and Ben Athiwaratkun and Manan Roongta and Ce Zhang and Li Erran Li and Raluca Ada Popa and Koushik Sen and Ion Stoica},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33}},
  note={Notion Blog},
  year={2025}
}
```