import hydra
import os
import logging
from omegaconf import OmegaConf

from rllm.agents.tool_agent import ToolAgent
from rllm.data import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.train.agent_trainer import AgentTrainer
from rllm.rewards.search_reward import rllm_reward_fn_search_boxed
from rllm.agents.system_prompts import SEARCH_SYSTEM_PROMPT

@hydra.main(config_path="../../rllm/train/config", config_name="ppo_trainer", version_base=None)
def main(config):
    OmegaConf.set_struct(config, False)
            
    config.actor_rollout_ref.model.path = "Qwen/Qwen3-4B"
    
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 32
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    config.actor_rollout_ref.actor.loss_agg_mode = "seq-mean-token-sum"
    
    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = 1
    
    config.actor_rollout_ref.rollout.enable_log_prob = False
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
    
    config.critic.model.path = "Qwen/Qwen3-4B"
    config.critic.ppo_micro_batch_size_per_gpu = 1
    config.critic.loss_agg_mode = "seq-mean-token-sum"
    
    config.reward_model.model.input_tokenizer = "Qwen/Qwen3-4B"
    config.reward_model.micro_batch_size_per_gpu = 1
    
    config.data.max_response_length = 8192
    
    config.trainer.logger = ["console"]
    
    config.trainer.n_gpus_per_node = 1
    
    config.env.name = "tool"
    config.env.env_args.max_steps = 20
    config.env.env_args.tools = ["local_search"]
    config.env.env_args.retrieval_server_url = os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:9000")
    
    config.agent.name = "tool_agent"
    config.agent.max_steps = 20
    config.agent.async_engine = False
    config.agent.agent_args = {}

    # Re-enable struct mode for safety
    OmegaConf.set_struct(config, True)

    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("hotpotqa_combined", "train")
    val_dataset = DatasetRegistry.load_dataset("hotpotqa_combined", "test")

    env_args = {
        "max_steps": 20, 
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