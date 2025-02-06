set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$HOME/data/deepscaler/test.parquet \
    data.output_path=$HOME/aime.parquet \
    data.n_samples=16 \
    data.batch_size=1024 \
    model.path=$HOME/rllm/global_step_480 \
    rollout.temperature=0.6 \
    rollout.response_length=32000 \
    rollout.top_k=-1 \
    rollout.top_p=0.95 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.tensor_model_parallel_size=1 $@
