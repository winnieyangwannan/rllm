#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --time=7-00:00:00

set -x
SLURM_CPUS_PER_TASK=12
SLURM_GPUS_PER_TASK=1
# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray stop --force
srun --time=7-00:00:00 --nodes=1 --ntasks=1 -w "$head_node" \
    env VLLM_ATTENTION_BACKEND=FLASH_ATTN ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray stop --force
    srun --time=7-00:00:00 --nodes=1 --ntasks=1 -w "$node_i" \
        env VLLM_ATTENTION_BACKEND=FLASH_ATTN ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --block &
    sleep 5
done

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

python3 -m verl.trainer.main_ppo_pipeline \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/home/tianjunz/rllm/rllm/data/train.parquet \
    data.val_files=/data/home/tianjunz/rllm/rllm/data/aime.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=30 \
    data.max_prompt_length=1024 \
    data.max_response_length=16000\
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.async_engine=True \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.vllm_log_prob=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-o3' \
    trainer.experiment_name='test' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.total_epochs=30 "${@:1}"

