set -x

export MINIWOB_URL="file:///home/tianjunz/miniwob-plusplus/miniwob/html/miniwob/"
# Also add more test to pipelined agent and finish
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# use_remove_padding has to be false because attention masks more than padding
python3 -m verl.trainer.main_ppo_agent \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/rllm-miniwob/train.parquet \
    data.val_files=$HOME/data/rllm-miniwob/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=8 \
    data.max_prompt_length=23000 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.async_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='7b-ppo-miniwob_agent' \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    env.name=browsergym \
    env.subtask=miniwob \
    env.miniwob_url="$MINIWOB_URL" \
    agent.name=webagent \
    agent.trajectory_episode_len=10 \
    agent.safe_batch_size=64
