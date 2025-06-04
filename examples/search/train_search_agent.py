import hydra

from rllm.agents.tool_agent import ToolAgent
from rllm.data import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.train.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.train.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("search_r1_combined", "train")
    val_dataset = DatasetRegistry.load_dataset("search_r1_combined", "test")

    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main() 