import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Tuple

from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.agents.agent import BaseAgent, Step, Trajectory
from rllm.parser import get_tool_parser
from rllm.tools.multi_tool import MultiTool
from rllm.agents.system_prompts import TOOL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class ToolAgent(BaseAgent):
    """
    An tool agent that can use tools to interact with the environment,
    refactored to follow the BaseAgent abstraction.
    """
    def __init__(self, model_name="", parser_name="qwen", tools=[]):
        """
        Initialize the ToolAgent.
        
        Args:
            model_name: Name of the model to use.
            parser_name: Name of the parser to use for tool calls.
            tools: List of tools available to the agent.
        """
        self.system_prompt = TOOL_SYSTEM_PROMPT
        self.tools = MultiTool(tools)
        parser_class = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class(model=model_name)
        self.model_name = model_name # Store model name if needed later

        self.tools_prompt = self.tool_parser.get_tool_prompt(
            json.dumps(self.tools.json, indent=2)
        )
        
        # Initialize state according to BaseAgent
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0
        self.reset() # Call reset to set initial state

    def _format_observation_as_messages(self, obs: Any) -> List[Dict]:
        """Helper to format observation into messages."""
        messages = []
        if isinstance(obs, dict):
            if 'question' in obs:
                messages.append({
                    "role": "user", 
                    "content": obs['question']
                })
            elif 'tool_outputs' in obs:
                # Format tool outputs from environment observation
                for tool_call_id, tool_output_str in obs['tool_outputs'].items():
                    messages.append({
                        "role": "tool", 
                        "content": tool_output_str, 
                        "tool_call_id": tool_call_id
                    })
        elif isinstance(obs, str):
            messages.append({
                "role": "user", 
                "content": obs
            })
        elif obs:
            messages.append({
                "role": "user", 
                "content": str(obs)
            })
            
        return messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict, **kwargs):
        """
        Updates the agent's state based on environment feedback.
        Formats observation and updates the trajectory.
        """
        # Update the previous step with results from the action
        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info

        # Format the observation for the next model call
        obs_messages = self._format_observation_as_messages(observation)
        self.messages.extend(obs_messages)

        if done:
            return
        
        # Create the new step for the current observation if trajectory not done.
        current_step = Step(
            observation=observation,
            step=self.step
        )
        self._trajectory.steps.append(current_step)

    def update_from_model(self, response: Any, **kwargs):
        """
        Updates the agent's state based on the model's response.
        Parses the response, updates messages, and the current step in the trajectory.
        """
        tool_calls_dict = {}
        assistant_content = ''

        # Process response (either string or OpenAI completion object)
        if isinstance(response, str):
            assistant_content = response
            # Attempt to parse tool calls from string response
            try:
                parsed_action = self.tool_parser.parse_input(response)
                if parsed_action:
                    tool_calls_dict = parsed_action.to_dict() # Assuming parser returns an object with to_dict()
            except Exception as e:
                logger.error(f"Failed to parse tool calls from string response: {e}")
                tool_calls_dict = {} # Indicate no valid tool calls parsed
        else: # Assuming OpenAI-like completion object
            completion = response
            assistant_content = completion.choices[0].message.content
            if completion.choices[0].message.tool_calls:
                tool_calls_dict = [{
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments # Arguments are already string here
                    }
                } for tool_call in completion.choices[0].message.tool_calls]

        # "Finishing tool call, calls the "finish" function, if no tools are found.
        if not tool_calls_dict:
            tool_calls_dict = {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": "finish",
                    "arguments": {
                        "response": response,
                    }
                }
            }
        # Append assistant message to chat history
        assistant_message = {"role": "assistant", "content": assistant_content}
        if tool_calls_dict:
             # Ensure arguments within tool_calls_dict are strings if needed by downstream processing
            for call in tool_calls_dict:
                if isinstance(call.get("function", {}).get("arguments"), dict):
                    call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
            assistant_message["tool_calls"] = tool_calls_dict
        self.messages.append(assistant_message)

        # Update the current step in the trajectory
        if self._trajectory.steps:
            current_step = self._trajectory.steps[-1]
            current_step.action = tool_calls_dict # Action is the list of tool calls
            current_step.model_response = response # Store raw response
            current_step.thought = ''
        else:
             logger.warning("update_from_model called before update_from_env after reset. Creating initial step.")
             current_step = Step(action=tool_calls_dict, model_response=response, step=self.step)
             self._trajectory.steps.append(current_step)

        self.step += 1
        
    def reset(self):
        """Resets the agent's state for a new episode."""
        self._trajectory = Trajectory()
        self.messages = [
            {"role": "system", "content": self.system_prompt + self.tools_prompt}
        ]
        self.step = 0
    
    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Returns the current message history for the model."""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory recorded so far."""
        return self._trajectory
    
    def get_current_state(self) -> Step:
        """Returns the last step added to the trajectory."""
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]


# Keep the __main__ block for testing if needed, update agent interaction if necessary
if __name__ == "__main__":
    # Create the environment (no batch_size parameter)
    envs = [ToolEnvironment(tools=["google_search"]), ToolEnvironment(tools=["google_search"])]
    import pdb; pdb.set_trace()
    # Create the batch agent with the tool agent
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    # Note: The engine now likely interacts via update_from_env/update_from_model
    # The sampling_params might need adjustment depending on the engine implementation
    # Specifically, how 'tools' are passed might change if the engine handles it internally

    sampling_params = {
        # "model": "gpt-4o", # Model name should likely be consistent with agent init
        "temperature": 0.6,
        "max_tokens": 8192, # Adjusted to match example, consider reducing if too large
        "top_p": 0.95,
        # "stop": ["```\n\n"], # Stop tokens if needed
        # The engine might need the tools directly, or get them from the agent/env
        "tools": envs[0].tools.json, 
    }

    # Initialize agents using the refactored class
    agents = [
        ToolAgent(tools=envs[0].tools.tools, model_name="gpt-4o", parser_name='openai'), 
        ToolAgent(tools=envs[1].tools.tools, model_name="gpt-4o", parser_name='openai')
    ]
    
    # Assuming AsyncAgentExecutionEngine is adapted for the BaseAgent interface
    from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine

    async_agent_execution_engine = AsyncAgentExecutionEngine(
        agents=agents,
        engine_name="openai", 
        envs=envs,
        tokenizer=tokenizer,  # Using transformers tokenizer if needed by engine
        # rollout_engine=None, # Pass if required
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

    # Adapt task format if needed by the engine or agent's initial observation handling
    # Assuming the engine passes the dict directly as the first observation
    tasks_formatted = [{"question": task} for task in tasks] 

    # Run the environment interaction
    asyncio.run(async_agent_execution_engine.execute_tasks(tasks_formatted))