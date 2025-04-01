
import json
from rllm.tools.multi_tool import MultiTool
from typing import List, Dict
from rllm.environments.batch_env import BatchedEnv

from typing import Any, Tuple

class ToolEnvironment:
    """
    A simple environment for tool-based agents that provides questions and evaluates responses.
    """
    
    def __init__(self, questions=None, tools = [], max_steps=10):
        self.batch_size = 1  # Always fixed to 1
        self.default_questions = [
            "What is the current stock price of Apple?",
            "What is the weather like in New York today?",
            "Who won the last Super Bowl?",
            "What are the top news headlines today?",
        ]
        self.questions = questions if questions else self.default_questions
        self.current_question = None
        self.step_count = 0
        self.max_steps = max_steps

        self.tools = MultiTool(tools)
    
    def reset(self, seed=None):
        """Reset the environment and return initial observations."""
        import random
        if seed is not None:
            random.seed(seed)
        
        # Select a random question
        self.current_question = random.choice(self.questions)
        self.step_count = 0
        
        # Return a single observation in a list to maintain the batch structure
        return {"question": self.current_question}, {}
    
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
        terminated = False
        truncated = self.step_count >= self.max_steps

        next_obs = None

        if action:
            tool_outputs = self._execute_tool_calls(action)
            next_obs = {"tool_outputs": tool_outputs}
        else:
            terminated = True
        # Return results as lists with single items to maintain batch structure
        return next_obs, reward, terminated, truncated, {"response": action}
    

    def _execute_tool_calls(self, tool_calls_dict):
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
        for idx, tool_call in enumerate(tool_calls_dict):
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
        seeds, 
        sizes,
        ps,
    ):
        self.envs = []
        self._env_id = []
        for i in range(batch_size):
            seed = seeds[i]
            size = sizes[i]
            p = ps[i]
            self.envs.append(ToolEnvironment(size=size, seed=seed, p=p))
            self._env_id.append(f"{seed}-{size}-{p}")

        self._batch_size = batch_size

    @property
    def env_id(self) -> List[str]:
        return self._env_id

    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    def reset(self, seed=0) -> Tuple[List, List]:
        observations = []
        for i, env in enumerate(self.envs):
            obs = env.reset(reset_map=False)
            observations.append(obs)
        return observations, [{}] * self.batch_size


    def step(self, actions: List[Any], env_idxs: List[int]=[]) -> Tuple[List, List, List, List, List]:

        if not env_idxs:
            assert len(actions) == self.batch_size, "Number of actions must match batch size"
            env_idxs = list(range(len(actions)))

        assert len(actions) == len(env_idxs), f"Number of actions ({len(actions)}) must match the env used {len(env_idxs)}"

        observations, rewards, terminateds, truncateds, infos = [], [], [], [], []
        # Send step command with actions
        for i, env_idx in enumerate(env_idxs):
            obs, reward, done, info = self.envs[env_idx].step(actions[i])
            observations.append(obs),
            rewards.append(reward)
            terminateds.append(done)
            truncateds.append(False)
            infos.append(info)


        return (observations, rewards, terminateds, 
                truncateds, infos)


    def close(self):
        return

    @staticmethod
    def from_extra_infos(extra_infos: List[Dict]) -> "BatchToolEnv":
        print(extra_infos)
        return BatchToolEnv(len(extra_infos), [0] * len(extra_infos), [0] * len(extra_infos), [0] * len(extra_infos))
        # seeds = [
        #     i["seed"] for i in extra_infos
        # ]
        # sizes = [
        #     i["size"] for i in extra_infos
        # ]
        # ps = [
        #     i["p"] for i in extra_infos
        # ]

        # return BatchToolEnv(batch_size=len(seeds), seeds=seeds, sizes=sizes, ps=ps)