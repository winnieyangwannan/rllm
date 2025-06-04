import ray
from typing import Type, Dict, Any, Optional, Union, List

from rllm.train.train_agent_ppo import main_task
from rllm.data import Dataset


class AgentTrainer:
    """
    A wrapper class that allows users to easily train custom agents with custom environments
    without having to directly interact with the underlying training infrastructure.
    """
    
    def __init__(
        self,
        agent_class: Type,
        env_class: Type,
        config: Optional[Union[Dict[str, Any], List[str]]] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ):
        """
        Initialize the AgentTrainer.
        
        Args:
            agent_class: The custom agent class to use for training
            env_class: The custom environment class to use for training
            config: Configuration overrides to apply to the default config
                   Can be a dictionary with dot notation keys (e.g., {"data.train_batch_size": 8})
                   or a list of strings in the format "key=value" (e.g., ["data.train_batch_size=8"])
        """
        self.agent_class = agent_class
        self.env_class = env_class

        self.config = config

        if train_dataset is not None:
            self.config.data.train_files = train_dataset.get_data_path()
        if val_dataset is not None:
            self.config.data.val_files= val_dataset.get_data_path()


    def train(self):
        if not ray.is_initialized():
            ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

        ray.get(main_task.remote(self.config, None, self.env_class, self.agent_class))