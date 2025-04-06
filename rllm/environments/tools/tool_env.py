
import json
from rllm.tools.multi_tool import MultiTool
from typing import List, Dict
from rllm.environments.batch_env import BatchedEnv

from typing import Any, Tuple, Optional

class ToolEnvironment:
    """
    A simple environment for tool-based agents that provides questions and evaluates responses.
    """
    
    def __init__(self, task: Optional[Dict] = None, tools: List[str] = [], max_steps=10):
        self.step_count = 0
        self.max_steps = max_steps

        self.tools = MultiTool(tools)
        self.task = task

        self.current_data = None
    
    def reset(self, task=None, seed=None):
        """Reset the environment and return initial observations."""
        import random
        if seed is not None:
            random.seed(seed)

        self.step_count = 0
        
        # Use the provided task if available, otherwise use the default task
        if task is not None:
            self.task = task
        
        # Return a single observation in a list to maintain the batch structure
        return {"question": self.task["question"]}, {}
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            actions: List containing a single action string from the agent
            
        Returns:
            next_observations, rewards, terminateds, truncateds, infos
        """
        self.step_count += 1
        
        reward = 0
        
        # Check if we should terminate
        terminated = self.step_count >= self.max_steps
        truncated = False

        next_obs = {}

        if action is not None and isinstance(action, list):
            tool_calls = action
            tool_outputs = self._execute_tool_calls(tool_calls)
            next_obs = {"tool_outputs": tool_outputs}
        else:
            terminated = True

        # Return results as lists with single items to maintain batch structure
        return next_obs, reward, terminated, truncated, {"response": action}
    

    def _execute_tool_calls(self, tool_calls: List[Dict]):
        import threading
        import queue

        # Create a dictionary to store results in order
        tool_outputs = {}
        output_queue = queue.Queue()
        threads = []

        def execute_tool(tool_call):
            tool_name = tool_call['function']['name']
            tool_args = json.loads(tool_call['function']['arguments'])
            tool_output = self.tools(tool_name=tool_name, **tool_args)

            tool_output_str = tool_output.output
            if isinstance(tool_output_str, (dict, list)):
                tool_output_str = json.dumps(tool_output_str)

            # tool_output_str = self.tool_parser.parse_output(tool_output)
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
    

class BatchToolEnv(BatchedEnv):
    def __init__(
        self,
        batch_size,
        batch_tasks,
    ):
        self.envs = []
        self._env_id = []
        for i in range(batch_size):
            tasks = batch_tasks[i]
            self.envs.append(ToolEnvironment(task=tasks))
            self._env_id.append(str(hash(tuple(tasks))))

        self._batch_size = batch_size

    @property
    def env_id(self) -> List[str]:
        return self._env_id

    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    def reset(self, seed=0) -> Tuple[List, List]:
        observations = []
        infos = []
        for i, env in enumerate(self.envs):
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return observations, infos

    def step(self, actions: List[Any], env_idxs: List[int]=[]) -> Tuple[List, List, List, List, List]:

        if not env_idxs:
            assert len(actions) == self.batch_size, "Number of actions must match batch size"
            env_idxs = list(range(len(actions)))

        assert len(actions) == len(env_idxs), f"Number of actions ({len(actions)}) must match the env used {len(env_idxs)}"

        observations, rewards, terminateds, truncateds, infos = [], [], [], [], []
        # Send step command with actions
        for i, env_idx in enumerate(env_idxs):
            obs, reward, done, truncated,info = self.envs[env_idx].step(actions[i])
            observations.append(obs),
            rewards.append(reward)
            terminateds.append(done)
            truncateds.append(truncated)
            infos.append(info)

        return (observations, rewards, terminateds, 
                truncateds, infos)

    def close(self):
        return

    @staticmethod
    def from_extra_infos(extra_infos: List[Dict]) -> "BatchToolEnv":
        batch_tasks = [i['task'] for i in extra_infos]
        return BatchToolEnv(batch_size=len(batch_tasks), batch_tasks=batch_tasks)