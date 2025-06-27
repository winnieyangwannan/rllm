from torch.distributed.device_mesh import init_device_mesh

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, create_sft_dataset
from verl.utils import hf_tokenizer
from verl.utils.device import get_device_name
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local


class AgentSFTTrainer(FSDPSFTTrainer):
    def __init__(self, config):
        # Initialize distributed training
        device_name = get_device_name()
        local_rank, rank, world_size = initialize_global_process_group()
        device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        dp_size = world_size // config.ulysses_sequence_parallel_size
        ulysses_device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))

        # Build tokenizer and datasets
        local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
        train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
        val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

        # Initialize parent class
        super().__init__(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)

    def _extract_messages_from_trajectory(self, traj):
        for attr_path in ["chat_completions", "trajectory.chat_completions", "trajectory[-1].chat_completions", "steps[0].chat_completions"]:
            try:
                obj = traj
                for attr in attr_path.split("."):
                    if "[" in attr:  # Handles "steps[0]"
                        attr_name, idx = attr.split("[")
                        idx = int(idx.rstrip("]"))
                        obj = getattr(obj, attr_name)[idx]
                    else:
                        obj = getattr(obj, attr)
                if obj:
                    return obj
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
        return None

    def process_trajectories(self, trajectories: list, reward_threshold: float):
        """Process trajectories into SFT format."""
        sft_data = []

        for traj in trajectories:
            if not traj:
                continue

            # Get reward from possible locations
            reward = getattr(traj, "reward", None) or getattr(getattr(traj, "trajectory", None), "reward", None) or getattr(getattr(traj, "steps", [None])[-1] if getattr(traj, "steps", None) else None, "reward", None)

            if not reward or reward < reward_threshold:
                continue

            # Extract and clean messages
            messages = self._extract_messages_from_trajectory(traj)
            if not messages:
                continue

            clean_messages = [{"role": msg["role"], "content": str(msg["content"]).strip()} for msg in messages if isinstance(msg, dict) and msg.get("role") and msg.get("content", "").strip()]

            if len(clean_messages) >= 2:
                sft_data.append({"messages": clean_messages})

        print(f"Processed {len(trajectories)} trajectories -> {len(sft_data)} valid examples")
        return sft_data

    def train(self):
        """Start training."""
        self.fit()
