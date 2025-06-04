
import hydra


from rllm.agents.math_agent import MathAgent
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.train.agent_trainer import AgentTrainer
from rllm.data import Dataset


def math_process_fn(example, idx: int):
    return {
        "problem": example.get("problem", ""),
        "tests": example.get("tests", []),
        "data_source": "livecodebench",
        "metadata": example.get("metadata", {})
    }

def prepare_dataset():
    train_dataset = Dataset(dataset_name="AIME", split="train", load_from_hf=False)
    val_dataset = Dataset(dataset_name="AIME", split="val", load_from_hf=False)
    return train_dataset, val_dataset

@hydra.main(config_path="pkg://rllm.train.config", config_name="ppo_trainer", version_base=None)
def main(config):
    from pprint import pprint
    pprint(config)
    trainer = AgentTrainer(
        agent_class=MathAgent,
        env_class=SingleTurnEnvironment,
        config=config,
    )
    trainer.train()

if __name__ == "__main__":
    main()