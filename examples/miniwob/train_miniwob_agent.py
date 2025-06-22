import os

import hydra

from rllm.agents.miniwob_agent import MiniWobAgent
from rllm.data import DatasetRegistry
from rllm.environments.browsergym.browsergym import BrowserGymEnv
from rllm.train.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.train.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("miniwob", "train")
    val_dataset = DatasetRegistry.load_dataset("miniwob", "test")

    env_args = {
        "subtask": "miniwob",
        "miniwob_url": os.getenv("MINIWOB_URL"),
    }

    trainer = AgentTrainer(
        agent_class=MiniWobAgent,
        env_class=BrowserGymEnv,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
