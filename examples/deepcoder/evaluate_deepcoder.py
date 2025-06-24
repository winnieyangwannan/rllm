import asyncio
import json
import os

from prepare_deepcoder_data import prepare_deepcoder_data
from transformers import AutoTokenizer

from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import code_reward_fn
from rllm.utils import save_trajectories


async def evaluate_deepcoder_on_livecodebench(
    model_name="agentica-org/DeepCoder-14B-Preview",
    num_examples=50,
    max_tokens=64000, 
    temperature=0.6,  
    top_p=0.95,       
    n_parallel_agents=1
):
    """
    Evaluate DeepCoder model on LiveCodeBench test generation dataset using local model.
    
    Args:
        model_name: Model name (for tokenizer)
        num_examples: Number of examples to evaluate
        max_tokens: Maximum tokens for generation (64K recommended)
        temperature: Sampling temperature (0.6 recommended)
        top_p: Top-p sampling parameter (0.95 recommended)
        n_parallel_agents: Number of parallel agents for evaluation
    """
    
    print(f"Evaluating {model_name} on LiveCodeBench test generation...")
    print("Using local model at http://localhost:30000/v1")
    
    print("Preparing LiveCodeBench test generation dataset...")
    test_dataset = prepare_deepcoder_data(test_size=10)
    tasks = test_dataset.get_data()
    
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "model": model_name,
    }
    
    engine = AgentExecutionEngine(
        agent_class=CompetitionCodingAgent,
        agent_args={}, 
        env_class=SingleTurnEnvironment,
        env_args={"reward_fn": code_reward_fn},
        rollout_engine=None,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=max_tokens,
        max_prompt_length=24576,  # Conservative increase for long coding problems (24K)
        config=None,
        n_parallel_agents=n_parallel_agents,
    )
    
    print(f"Starting evaluation with {len(tasks)} tasks and {n_parallel_agents} parallel agents...")
    
    # Execute tasks
    results = await engine.execute_tasks(tasks)

    # Save trajectories for further analysis
    trajectories_file = f"deepcoder_trajectories_{num_examples}.pt"
    save_trajectories(results, filename=trajectories_file)
    print(f"Trajectories saved to: {trajectories_file}")
    
    return results


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    results = asyncio.run(evaluate_deepcoder_on_livecodebench(
        model_name="agentica-org/DeepCoder-14B-Preview",
        num_examples=len(prepare_deepcoder_data().get_data()), 
        max_tokens=64000, 
        temperature=0.6,  
        top_p=0.95,       
        n_parallel_agents=64
    )) 