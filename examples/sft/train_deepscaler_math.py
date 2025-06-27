"""
Usage:
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 train_deepscaler_math.py \
        model.partial_pretrain=Qwen/Qwen2.5-1.5B-Instruct \
        model.trust_remote_code=true \
        trainer.total_epochs=1 \
        data.train_batch_size=2 \
        data.micro_batch_size_per_gpu=1 \
        data.max_length=8192 \
        data.truncation=right \
        data.multiturn.enable=true \
        data.multiturn.messages_key=messages \
        data.train_files=large_sft_data.parquet \
        data.val_files=sft_data.parquet \
        trainer.default_local_dir=outputs/simple_sft_test
"""

import hydra
from agent_sft_trainer import AgentSFTTrainer
from omegaconf import DictConfig


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="sft_trainer", version_base=None)
def main(config: DictConfig):
    # initialize trainer
    trainer = AgentSFTTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
