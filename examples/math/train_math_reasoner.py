
import hydra

from rllm.agents.math_agent import MathAgent
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.train.agent_trainer import AgentTrainer

from .run_math_reasoner import prepare_math_data


@hydra.main(config_path="pkg://rllm.train.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset, test_dataset = prepare_math_data()
    
    trainer = AgentTrainer(
        agent_class=MathAgent,
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()