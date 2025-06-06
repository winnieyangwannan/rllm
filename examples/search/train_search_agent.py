import hydra

from rllm.agents.tool_agent import ToolAgent
from rllm.data import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.train.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.train.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("search_r1_combined", "train")
    val_dataset = DatasetRegistry.load_dataset("search_r1_combined", "test")

    # Example agent and environment arguments
    agent_args = {
        "temperature": 0.7,
        "max_tokens_per_step": 512,
        # Add other agent-specific arguments as needed
    }
    
    env_args = {
        "max_turns": 5,
        "timeout": 30,
        # Add other environment-specific arguments as needed
    }

    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_args=agent_args,
        env_args=env_args,
    )
    trainer.train()

if __name__ == "__main__":
    main()