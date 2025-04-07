import asyncio
import json
import logging
from typing import Dict, List

from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.models.agent import BaseAgent
from rllm.parser import get_tool_parser
from rllm.tools.multi_tool import MultiTool

logger = logging.getLogger(__name__)


TOOL_SYSTEM_PROMPT= """
You are a tool agent. You are given a task to complete. You have a set of tools at your disposal. Before you use the tools, outputting your thoughts before calling the tools. 
"""

class ToolAgent(BaseAgent):
    """
    An tool agent that can use tools to interact with the environment.
    """
    def __init__(self, model_name="", parser_name="qwen", tools=[]):
        """
        Initialize the ToolAgent.
        
        Args:
            tool_caller: The tool caller object that handles tool execution
            system_prompt: The system prompt to use for the agent
        """
        # Use provided system prompt or default to TOOL_SYSTEM_PROMPT
        self.system_prompt = TOOL_SYSTEM_PROMPT

        self.trajectory = []
        self.tools = MultiTool(tools)
        parser_class = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class(model=model_name)

        self.tools_prompt = self.tool_parser.get_tool_prompt(json.dumps(self.tools.json, indent=2))
        self.messages = [
            {"role": "system", "content": self.system_prompt + self.tools_prompt}
        ]
        
    def _pre_get_action(self, trajectory: List[Dict]):
        self.messages.extend(self.format_observation_as_messages(trajectory[-1]['next_observation']))
        return self.messages
    
    def _post_get_action(self, response):
        if isinstance(response, str):
            self.messages.append({"role": "assistant", "content": response})
            tool_calls_dict = self.tool_parser.parse_input(response).to_dict()
            print("tool_calls_dict:", tool_calls_dict)
            return tool_calls_dict
        else:
            completion = response
            self.messages.append({"role": "assistant", "content": completion.choices[0].message.content})
            if completion.choices[0].message.tool_calls:
                tool_calls_dict = [{
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in completion.choices[0].message.tool_calls]

                self.messages[-1]["tool_calls"] = tool_calls_dict

                return tool_calls_dict

    def _execute_tool_calls(self, tool_calls_dict):      
        for tool_call in tool_calls_dict:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_output = self.tools(tool_name=tool_name, **tool_args)
            tool_output_str = self.parser.parse_output(tool_output)
            
            self.messages.append({
                "role": "tool",
                "content": tool_output_str,
                "tool_call_id": tool_call.id
            })
    
    def update(self, action, observation, next_observation, reward, terminated, truncated, info):
        # Store the step information in the trajectory
        step = {
            "observation": observation,
            "next_observation": next_observation,
            "reward": reward,
            "action": action,
            "response": action,  # In this agent, the action is the response
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
        self.trajectory.append(step)
    
    def reset(self):
        # Reset the agent's state
        self.trajectory = []
        self.messages = [{"role": "system", "content": self.system_prompt}]
    
    def compute_training_reward(self, trajectory):
        # Default implementation just returns the last reward
        if trajectory and len(trajectory) > 0:
            return trajectory[-1]["reward"]
        return 0.0
    
    def convert_observation_to_string(self, obs, with_system_prompt=False):
        if 'question' in obs:
            self.messages.append({"role": "user", "content": obs['question']})
        elif 'tool_outputs' in obs:
            for id, tool_output_str in obs['tool_outputs'].items():
                self.messages.append({"role": "tool", "content": tool_output_str, "tool_call_id": id})

        return obs
    
    def format_observation_as_messages(self, obs):
        messages = []
        if 'question' in obs:
            messages.append({"role": "user", "content": obs['question']})
        elif 'tool_outputs' in obs:
            for id, tool_output_str in obs['tool_outputs'].items():
                messages.append({"role": "tool", "content": tool_output_str, "tool_call_id": id})

        return messages

if __name__ == "__main__":
    # Create the environment (no batch_size parameter)
    envs = [ToolEnvironment(tools=["google_search"]), ToolEnvironment(tools=["google_search"])]
    
    # Create the batch agent with the tool agent
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    # agent = ToolAgent(model_name="Qwen/Qwen2.5-1.5B-Instruct", tools=["google_search"])

    from rllm.rllm.models.async_agent_execution_engine import AsyncAgentExecutionEngine

    sampling_params = {
        "model": "gpt-4o",
        "temperature": 0.6,
        "max_tokens": 8192,
        "top_p": 0.95,
        # "stop": ["```\n\n"],
        "tools": envs[0].tools.json,
    }

    agents = [ToolAgent(tools=["google_search"], model_name="Qwen/Qwen2.5-1.5B-Instruct"), ToolAgent(tools=["google_search"], model_name="Qwen/Qwen2.5-1.5B-Instruct")]
    
    async_agent_execution_engine = AsyncAgentExecutionEngine(
        agents=agents,
        engine_name="openai", 
        envs=envs,
        tokenizer=tokenizer,  # Using transformers tokenizer
        rollout_engine=None,
        sampling_params=sampling_params
    )

    tasks = [
        "Who won the 2024 Super Bowl and what was the final score?",
        "What is the current population of Tokyo in 2024?",
        "When is the next solar eclipse visible from North America?",
        "Who is the current CEO of OpenAI and when did they take the position?",
        "What was the highest grossing movie of 2023?",
        "What is the latest breakthrough in fusion energy research?",
    ]

    tasks = [{"question": task} for task in tasks]

    # Run the environment interaction
    asyncio.run(async_agent_execution_engine.execute_tasks(tasks))