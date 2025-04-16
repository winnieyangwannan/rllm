set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python3 -m verl.trainer.main_ppo_agent \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/rllm-frozenlake/train.parquet \
    data.val_files=$HOME/data/rllm-frozenlake/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=28000 \
    data.max_response_length=4000 \
    actor_rollout_ref.model.path="r2e-edits/qwen25coder-7b-instruct-end2end_sonnet_combined_maxstep40-rft-20k_bz8_epoch2_lr1en5-v2" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.async_engine=False \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.002 \
    trainer.critic_warmup=0 \
    trainer.rejection_sample=False \
    trainer.logger=['console'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='7b-ppo-frozenlake_agent' \
    +trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=200 \
    trainer.total_epochs=100 \
    env.name=sweenv \
    agent.name=sweagent \
    agent.max_trajectory_length=32000 \
    agent.max_episodes=40
