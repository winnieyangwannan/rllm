set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

python -m verl.trainer.main_trajectory \
    data.path=$HOME/data/rllm-miniwob/train.parquet \
    model.path=Qwen/Qwen2.5-7B-Instruct-1M \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    rollout.tensor_model_parallel_size=2 \
    rollout.name=vllm \
    rollout.gpu_memory_utilization=0.9 \
    rollout.n=1 \
    agent.trajectory_episode_len=2 \
    env.name=browsergym \
    env.subtask=miniwob \
    env.miniwob_url="$MINIWOB_URL" \
    agent.name=webagent \
    agent.trajectory_episode_len=20 \
    agent.safe_batch_size=32 \
    
    
    
