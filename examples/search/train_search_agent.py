import hydra
import os
import logging

from rllm.agents.tool_agent import ToolAgent
from rllm.data import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.train.agent_trainer import AgentTrainer
from rllm.rewards.search_reward import rllm_reward_fn_search_boxed
from rllm.agents.system_prompts import SEARCH_SYSTEM_PROMPT

@hydra.main(config_path="../../rllm/train/config", config_name="search_agent_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hotpotqa_combined", "train")
    val_dataset = DatasetRegistry.load_dataset("hotpotqa_combined", "test")

    env_args = {
        "max_steps": 3, 
        "tools": ["local_search"],
        "reward_fn": rllm_reward_fn_search_boxed,
    }
    
    agent_args = {
        "system_prompt": SEARCH_SYSTEM_PROMPT,
        "tools": ["local_search"],
        "parser_name": "qwen"
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