import asyncio
import json
import logging

from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.models.agent import BaseAgent
from rllm.tools.multi_tool import MultiTool

logger = logging.getLogger(__name__)


TOOL_SYSTEM_PROMPT= """
You are a tool agent. You are given a task to complete. You have a set of tools at your disposal. Before you use the tools, outputting your thoughts before calling the tools.
"""

class ToolAgent(BaseAgent):
    """
    An agent that can use tools to interact with the environment.
    This agent is compatible with OpenAI's tool calling format.
    """
    
    def __init__(self, system_prompt="", tools=[]):
        """
        Initialize the ToolAgent.
        
        Args:
            tool_caller: The tool caller object that handles tool execution
            system_prompt: The system prompt to use for the agent
        """
        self.tools = MultiTool(tools)

        # Use provided system prompt or default to TOOL_SYSTEM_PROMPT
        self.system_prompt = system_prompt if system_prompt else TOOL_SYSTEM_PROMPT

        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.trajectory = []
        self.sampling_params = {
            "model": "gpt-4o",
            "temperature": 0.6,
            "max_tokens": 8192,
            "top_p": 0.95,
            "stop": ["```\n\n"],
            "tools": self.tools.json,
        }
        
    def _pre_get_action(self, trajectory):
        if trajectory is None:
            return self.messages
        
        if 'question' in trajectory:
            self.messages.append({"role": "user", "content": trajectory['question']})
        elif 'tool_outputs' in trajectory:
            for id, tool_output_str in trajectory['tool_outputs'].items():
                self.messages.append({"role": "tool", "content": tool_output_str, "tool_call_id": id})
        
        return self.messages
    
    def _post_get_action(self, completion):
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
        pass


if __name__ == "__main__":
    
    # Create the environment (no batch_size parameter)
    env = ToolEnvironment(tools=["google_search"])
    
    # Create the batch agent with the tool agent
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("agentica-org/DeepScaleR-1.5B-Preview")

    from rllm.models.engine import AgentExecutionEngine
    
    batch_agent = AgentExecutionEngine(
        agent_class=ToolAgent,
        agent_args={"tools": ["google_search"]},
        engine_name="openai", 
        n_parallel_agents=1,  # Set to 1 to match environment's fixed batch size
        env=env,
        tokenizer=tokenizer,  # Using transformers tokenizer
        rollout_engine=None
    )
    # Run the environment interaction
    asyncio.run(batch_agent.interact_environment_async())