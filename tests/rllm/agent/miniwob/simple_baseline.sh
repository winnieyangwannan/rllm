set -x

python -m verl.trainer.main_trajectory \
    data.path=$HOME/data/rllm-miniwob/test.parquet \
    data.n_samples=1 \
    data.batch_size=8 \
    data.output_metric_path=./simple_baseline_result/result.csv \
    model.path=Qwen/Qwen2.5-7B-Instruct-1M \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    rollout.tensor_model_parallel_size=4 \
    rollout.name=vllm \
    rollout.temperature=0.6 \
    rollout.prompt_length=8192 \
    rollout.response_length=128 \
    rollout.gpu_memory_utilization=0.3 \
    rollout.async_engine=False \
    rollout.log_prob_micro_batch_size_per_gpu=1 \
    rollout.n=1 \
    env.name=browsergym \
    env.subtask=miniwob \
    env.miniwob_url="$MINIWOB_URL" \
    agent.name=webagent \
    agent.max_episodes=10
    
    
