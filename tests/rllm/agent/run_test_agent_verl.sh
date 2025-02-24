set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

python test_agent_verl.py \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    rollout.tensor_model_parallel_size=2 \
    rollout.name=vllm \
    rollout.gpu_memory_utilization=0.9 \
    rollout.n=1 \
    
