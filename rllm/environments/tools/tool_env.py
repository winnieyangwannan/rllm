import json
import warnings
from typing import Dict, List, Optional, Union

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward
from rllm.tools.multi_tool import MultiTool


class ToolEnvironment(BaseEnv):
    """
    A simple environment for tool-based agents that provides questions and evaluates responses.
    """
    
    def __init__(self, task: Optional[Dict] = None, tools: List[str] = [], reward_fn: Optional[RewardFunction] = None, max_steps=10):
        self.step_count = 0
        self.max_steps = max_steps

        self.tools = MultiTool(tools)
        self.task = task
        self.reward_fn = reward_fn
        if reward_fn is None:
            warnings.warn("No reward function specified, will get 0 reward.")
            self.reward_fn = zero_reward
    
    def reset(self):
        """Reset the environment and return initial observations."""
        self.step_count = 0
        
        return self.task, {}
    
    def step(self, action: Union[List[Dict], str, Dict]):
        """
        Take a step in the environment based on the action.
        
        Args:
            actions: List containing a single action string from the agent
            
        Returns:
            next_observations, rewards, terminateds, infos
        """
        if isinstance(action, dict):
            action = [action]
        self.step_count += 1
        
        reward = 0
        # Check if we should terminate
        done = self.step_count >= self.max_steps or isinstance(action, str)
        # Check if action contains a "finish" tool call
        if isinstance(action, list) and action:
            for tool_call in action:
                if tool_call.get('function', {}).get('name') == 'finish':
                    done = True
                    break
        if done:
            # Cannot find tool calls which means the agent is not using the tool and is done.
            if isinstance(action, str):
                llm_response = action
            elif isinstance(action, list):
                # Find the finish tool call
                finish_action = None
                for tool_call in action:
                    if tool_call.get('function', {}).get('name') == 'finish':
                        finish_action = tool_call
                        break
                arguments = finish_action.get('function', {}).get('arguments', {})
                llm_response = arguments.get('response', '')
            
            reward_output = self.reward_fn(task_info=self.task, action=llm_response)
            return {}, reward_output.reward, done, {"response": action, "metadata": reward_output.metadata}

        tool_calls = action
        tool_outputs = self._execute_tool_calls(tool_calls)
        next_obs = {"tool_outputs": tool_outputs}

        # Return results as lists with single items to maintain batch structure
        return next_obs, reward, done, {"response": action, "metadata": {}}
    
    def _execute_tool_calls(self, tool_calls: List[Dict]):
        import queue
        import threading

        # Create a dictionary to store results in order
        tool_outputs = {}
        output_queue = queue.Queue()
        threads = []

        def execute_tool(tool_call):
            tool_name = tool_call['function']['name']
            tool_args = json.loads(tool_call['function']['arguments'])
            tool_output = self.tools(tool_name=tool_name, **tool_args)
            tool_output_str = tool_output.to_string()

            output_queue.put((tool_call['id'], tool_output_str))

        # Create and start a thread for each tool call
        for idx, tool_call in enumerate(tool_calls):
            thread = threading.Thread(target=execute_tool, args=(tool_call,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results and store in order
        while not output_queue.empty():
            tool_call_id, output_str = output_queue.get()
            tool_outputs[tool_call_id] = output_str

        return tool_outputs
    
    @staticmethod
    def from_dict(env_args: Dict) -> "ToolEnvironment":
        tools = env_args.pop('tools', [])
        reward_fn = env_args.pop('reward_fn', None)
        max_steps = env_args.pop('max_steps', 10)
        return ToolEnvironment(task=env_args, tools=tools, max_steps=max_steps, reward_fn=reward_fn)