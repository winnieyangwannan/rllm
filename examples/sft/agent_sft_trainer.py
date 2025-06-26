import sys

# Add verl to path
sys.path.append("../../verl")

from torch.distributed.device_mesh import init_device_mesh

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, create_sft_dataset
from verl.utils import hf_tokenizer
from verl.utils.device import get_device_name
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local


class AgentSFTTrainer(FSDPSFTTrainer):
    def __init__(self, config):
        print("Initializing AgentSFTTrainer...")

        # Initialize distributed training
        device_name = get_device_name()
        local_rank, rank, world_size = initialize_global_process_group()

        device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

        dp_size = world_size // config.ulysses_sequence_parallel_size
        ulysses_device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))

        # Build tokenizer and datasets
        print(f"Loading model and tokenizer: {config.model.partial_pretrain}")
        local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

        print("Loading datasets...")
        print(f" - Train: {config.data.train_files}")
        print(f" - Val: {config.data.val_files}")

        train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
        val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

        print(f" - Train dataset size: {len(train_dataset)}")
        print(f" - Val dataset size: {len(val_dataset)}")

        # Initialize parent class
        super().__init__(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)

        print("AgentSFTTrainer ready for training!")

    def train(self):
        """Start training process."""
        print("Starting SFT training...")
        print("Configuration:")
        print(f" - Model: {self.config.model.partial_pretrain}")
        print(f" - Epochs: {self.config.trainer.total_epochs}")
        print(f" - Batch size: {self.config.data.train_batch_size}")
        print(f" - Micro batch size: {self.config.data.micro_batch_size_per_gpu}")
        print(f" - Max length: {self.config.data.max_length}")
        print(f" - Truncation: {self.config.data.truncation}")

        # verl's training loop
        self.fit()

        print("Training completed!")
