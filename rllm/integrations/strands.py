import json
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any, TypeVar

from pydantic import BaseModel
from strands import Agent
from strands.models.model import Model
from strands.types.content import ContentBlock, Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec

from rllm.agents.agent import Step, Trajectory
from rllm.engine.rollout import ModelOutput, RolloutEngine

T = TypeVar("T", bound=BaseModel)


class RLLMModel(Model):
    """Model class that uses rLLM's RolloutEngine for inference."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        """Initialize the RLLMModel.

        Args:
            rollout_engine: The rLLM RolloutEngine instance to use for inference
        """
        self.rollout_engine = rollout_engine
        # Minimal config to satisfy strands Model interface
        model_id = kwargs.pop("model_id", None)
        self._config: dict[str, Any] = {"model_id": model_id, "params": dict(kwargs)}

    def update_config(self, **model_config: Any) -> None:
        if "model_id" in model_config:
            self._config["model_id"] = model_config.pop("model_id")
        params = self._config.get("params") or {}
        params.update(model_config)
        self._config["params"] = params

    def get_config(self) -> dict[str, Any]:
        return {"model_id": self._config.get("model_id"), "params": dict(self._config.get("params") or {})}

    async def structured_output(self, output_model: type[T], prompt: Messages, system_prompt: str | None = None, **kwargs: Any) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output from the model.

        Note: This is a basic implementation that converts the text response to the output model.
        For more advanced structured output, consider using the native OpenAI structured output features.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments.

        Yields:
            Model events with the last being the structured output.
        """
        # Convert Strands messages to chat completion format
        messages = self._convert_messages_to_chat_format(prompt, system_prompt)

        # Add instruction for structured output
        if messages and messages[-1]["role"] == "user":
            original_content = messages[-1]["content"]
            messages[-1]["content"] = f"{original_content}\n\nPlease respond with a JSON object that matches this schema: {output_model.model_json_schema()}"

        # Get response from rollout engine
        response_text = (await self.rollout_engine.get_model_response(messages, **kwargs)).text

        print(f"response_text: {response_text}")
        try:
            # Try to parse the response as JSON and convert to the output model
            import json

            # Extract JSON from response if it's wrapped in other text
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed_data = json.loads(json_str)
                structured_output = output_model(**parsed_data)
                yield {"output": structured_output}
            else:
                raise ValueError("No valid JSON found in response")

        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse structured output: {e}") from e

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream conversation with the model using RolloutEngine with tool execution loop.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments.

        Yields:
            Formatted message chunks from the model.
        """
        # Convert Strands messages to chat completion format
        chat_messages = self._convert_messages_to_chat_format(messages, system_prompt)
        
        # Prepare tools mapping for execution
        tools_map = {}
        tools_param = None
        
        # Try to get tools from the agent if available
        if hasattr(self, 'agent') and hasattr(self.agent, '_original_tools'):
            tools_map = self._build_tools_map(self.agent._original_tools)
            tools_param = self._convert_tool_specs_to_openai_format(self.agent._original_tools)
        elif tool_specs:
            tools_map = self._build_tools_map(tool_specs)
            tools_param = self._convert_tool_specs_to_openai_format(tool_specs)

        # Tool execution loop
        max_turns = 10  # Prevent infinite loops
        turn_count = 0
        
        while turn_count < max_turns:
            turn_count += 1
            
            # Call rollout engine
            call_kwargs = dict(kwargs)
            if tools_param:
                call_kwargs["tools"] = tools_param
                
            model_output: ModelOutput = await self.rollout_engine.get_model_response(chat_messages, **call_kwargs)
            
            # Check if we have tool calls to execute
            if getattr(model_output, "tool_calls", None):
                
                # Yield message start for assistant
                yield {"messageStart": {"role": "assistant"}}
                
                # Emit tool call events and execute tools
                tool_results = []
                for tc in model_output.tool_calls:
                    # Extract tool call info
                    tool_call_info = self._extract_tool_call_info(tc)
                    
                    # Yield tool call event
                    yield {"toolCall": tool_call_info}
                    
                    # Execute the tool
                    tool_result = await self._execute_tool(tool_call_info, tools_map)
                    tool_results.append(tool_result)
                
                # Yield message stop with tool_calls reason
                yield {"messageStop": {"stopReason": "tool_calls"}}
                
                # Add the assistant message with tool_calls first
                assistant_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["input"])
                            }
                        }
                        for tc in [tool_call_info]  # We're processing one at a time
                    ]
                }
                chat_messages.append(assistant_message)
                
                # Then add tool results
                chat_messages.extend(tool_results)
                
                # Continue the loop to get the final response
                continue
            else:
                # No tool calls, this is the final response
                # Yield message start
                yield {"messageStart": {"role": "assistant"}}
                
                # Yield content
                response_text = model_output.text or ""
                yield {"contentBlockStart": {"start": {}}}
                
                # Stream the response text
                chunk_size = 50
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i : i + chunk_size]
                    if chunk:
                        yield {"contentBlockDelta": {"delta": {"text": chunk}}}
                
                yield {"contentBlockStop": {}}
                
                # Determine stop reason
                stop_reason = getattr(model_output, "finish_reason", "end_turn")
                yield {"messageStop": {"stopReason": stop_reason}}
                
                return  # Exit the loop
                
        # If we reach here, we hit the max turns limit
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": "I apologize, but I've reached the maximum number of tool execution turns. Please try again."}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "max_turns"}}

    def _build_tools_map(self, tool_specs: list[ToolSpec]) -> dict:
        """Build a mapping from tool names to callable tools."""
        tools_map = {}
        for spec in tool_specs:
            try:
                # Get tool name
                name = None
                
                # Handle DecoratedFunctionTool objects (from strands_tools)
                if hasattr(spec, "tool_name"):
                    name = spec.tool_name
                elif hasattr(spec, "name"):
                    name = spec.name
                elif isinstance(spec, dict):
                    name = spec.get("name") or (spec.get("function") or {}).get("name")
                elif callable(spec):
                    # For plain functions, try to get name from function object
                    if hasattr(spec, "__name__"):
                        name = spec.__name__
                    elif hasattr(spec, "__qualname__"):
                        name = spec.__qualname__.split('.')[-1]
                
                if not name:
                    continue
                
                # Get the callable tool
                if callable(spec):
                    # The spec itself is callable
                    tools_map[name] = spec
                elif hasattr(spec, "function") and callable(spec.function):
                    # The spec has a callable function attribute
                    tools_map[name] = spec.function
                else:
                    continue
                
            except Exception as e:
                continue
        
        return tools_map

    def _convert_tool_specs_to_openai_format(self, tool_specs: list[ToolSpec]) -> list[dict]:
        """Convert tool specs to OpenAI tools format."""
        tools_param = []
        
        for spec in tool_specs:
            try:
                # Handle DecoratedFunctionTool objects (from strands_tools)
                if hasattr(spec, "tool_spec"):
                    tool_spec = spec.tool_spec
                    if isinstance(tool_spec, dict):
                        name = tool_spec.get("name")
                        input_schema = tool_spec.get("inputSchema", {})
                        if isinstance(input_schema, dict) and "json" in input_schema:
                            params = input_schema["json"]
                            if name and isinstance(params, dict):
                                tools_param.append({
                                    "type": "function",
                                    "function": {"name": name, "parameters": params},
                                })
                                continue
                
                # Prefer SDK helper if available (object specs)
                if hasattr(spec, "to_openai_tool"):
                    maybe = spec.to_openai_tool()
                    if isinstance(maybe, dict):
                        tools_param.append(maybe)
                        continue

                # Dict-shaped specs (common with strands-agents-tools)
                if isinstance(spec, dict):
                    # If already OpenAI-shaped, pass through
                    if spec.get("type") == "function" and isinstance(spec.get("function"), dict):
                        tools_param.append(spec)
                        continue
                    # Otherwise, synthesize OpenAI tool from common fields
                    name = spec.get("name") or (spec.get("function") or {}).get("name")
                    params = (
                        spec.get("parameters")
                        or spec.get("input_schema")
                        or spec.get("schema")
                        or ((spec.get("inputSchema") or {}).get("json") if isinstance(spec.get("inputSchema"), dict) else None)
                    )
                    if name and isinstance(params, dict):
                        tools_param.append({
                            "type": "function",
                            "function": {"name": name, "parameters": params},
                        })
                    continue

                # Object specs without helper: attempt attribute-based extraction
                name = getattr(spec, "name", None)
                params = getattr(spec, "input_schema", None) or getattr(spec, "parameters", None)
                if name and isinstance(params, dict):
                    tools_param.append({
                        "type": "function",
                        "function": {"name": name, "parameters": params},
                    })
            except Exception as e:
                print(f"[RLLMModel] Warning: Failed to convert tool spec {spec}: {e}")
                continue
                
        return tools_param

    def _extract_tool_call_info(self, tool_call) -> dict:
        """Extract tool call information from ModelOutput tool call."""
        try:
            func = tool_call.get("function", {}) if isinstance(tool_call, dict) else getattr(tool_call, "function", {})
            fname = func.get("name") if isinstance(func, dict) else getattr(func, "name", None)
            fargs_raw = func.get("arguments") if isinstance(func, dict) else getattr(func, "arguments", None)
            
            # Parse arguments if they're JSON strings
            fargs = {}
            if isinstance(fargs_raw, str):
                import json as _json
                try:
                    fargs = _json.loads(fargs_raw) if fargs_raw.strip() else {}
                except Exception:
                    fargs = {"_raw": fargs_raw}
            elif isinstance(fargs_raw, dict):
                fargs = fargs_raw
                
            tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
            
            return {"id": tool_id, "name": fname or "tool", "input": fargs}
        except Exception:
            return {"id": "unknown", "name": "unknown", "input": {}}

    async def _execute_tool(self, tool_call_info: dict, tools_map: dict) -> dict:
        """Execute a tool call and return the result as a chat message."""
        tool_name = tool_call_info["name"]
        tool_input = tool_call_info["input"]
        tool_id = tool_call_info["id"]
        
        try:
            # Find the tool
            if tool_name not in tools_map:
                result = f"Error: Tool '{tool_name}' not found"
            else:
                tool = tools_map[tool_name]
                
                # Execute the tool
                if callable(tool):
                    result = tool(**tool_input)
                else:
                    result = f"Error: Tool '{tool_name}' is not callable"
                
        except Exception as e:
            result = f"Error executing tool '{tool_name}': {str(e)}"
        
        # Return as OpenAI tool message format
        return {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": str(result)
        }

    def _convert_messages_to_chat_format(self, messages: Messages, system_prompt: str | None = None) -> list[dict[str, str]]:
        """Convert Strands messages to chat completion format.

        This reuses logic similar to OpenAIModel but outputs the simpler format expected by RolloutEngine.

        Args:
            messages: Strands messages to convert
            system_prompt: Optional system prompt to prepend

        Returns:
            List of chat completion messages
        """
        chat_messages = []

        # Add system prompt if provided
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})

        for message in messages:
            role = message["role"]
            content = message["content"]

            # Extract text content from Strands format
            text_content = ""
            for content_block in content:
                if "text" in content_block:
                    text_content += content_block["text"]
                elif "toolUse" in content_block:
                    # For now, represent tool use as text
                    tool_use = content_block["toolUse"]
                    text_content += f"[Tool: {tool_use['name']} with input: {tool_use.get('input', {})}]"
                elif "toolResult" in content_block:
                    # For now, represent tool result as text
                    tool_result = content_block["toolResult"]
                    text_content += f"[Tool Result: {tool_result.get('content', [])}]"
                # TODO: Handle other content types like images, documents if needed

            if text_content.strip():  # Only add if there's actual content
                chat_messages.append({"role": role, "content": text_content})

        return chat_messages


class StrandsAgent(Agent):
    def __init__(self, model: RLLMModel, **kwargs):
        """Initialize StrandsAgent with trajectory tracking.

        Args:
            **kwargs: Additional arguments to pass to the base Agent class
        """

        # Save tools before passing to base class
        self._original_tools = kwargs.get('tools', [])
        
        super().__init__(model=model, **kwargs)
        
        # Set agent reference in the model so it can access tools
        model.agent = self
        
        self._trajectory = Trajectory()
        self._current_step = None

    @property
    def trajectory(self) -> Trajectory:
        """Get the current trajectory object."""
        return self._trajectory

    @property
    def chat_completions(self):
        """Convert agent's messages into chat completions format."""
        completions = []
        for message in self.messages:
            # Convert Strands message format to chat completion format
            if isinstance(message.get("content"), list):
                # Handle multi-content messages
                text_content = ""
                for content_block in message["content"]:
                    if isinstance(content_block, dict) and "text" in content_block:
                        text_content += content_block["text"]
                completions.append({"role": message["role"], "content": text_content})
            else:
                # Handle simple string content
                completions.append({"role": message["role"], "content": str(message.get("content", ""))})
        return completions

    def reset_trajectory(self, task: Any = None):
        """Reset the trajectory for a new episode."""
        self._trajectory = Trajectory(task=task)
        self._current_step = None

    def _start_new_step(self, observation: Any = None):
        """Start a new step in the trajectory."""
        self._current_step = Step(chat_completions=self.chat_completions.copy(), observation=observation)

    def _finish_current_step(self, model_response: str = "", action: Any = None, reward: float = 0.0, done: bool = False):
        """Finish the current step and add it to the trajectory."""
        if self._current_step is not None:
            self._current_step.model_response = model_response
            self._current_step.action = action
            self._current_step.reward = reward
            self._current_step.done = done
            self._current_step.chat_completions = self.chat_completions.copy()

            self._trajectory.steps.append(self._current_step)
            self._trajectory.reward += reward
            self._current_step = None

    def _finish_current_step_from_result(self, result: Any):
        """Finish the current step by extracting info from AgentResult."""
        model_response = ""
        action = None

        if hasattr(result, "message") and result.message:
            # Extract text content from the final message
            if hasattr(result.message, "content") and isinstance(result.message.content, list):
                for content_block in result.message.content:
                    if isinstance(content_block, dict) and "text" in content_block:
                        model_response += content_block["text"]
            elif hasattr(result.message, "content"):
                model_response = str(result.message.content)

            action = result.message if hasattr(result, "message") else result

        # Determine if this step is done
        done = hasattr(result, "stop_reason") and result.stop_reason in ["end_turn", "stop_sequence"]

        self._finish_current_step(model_response=model_response, action=action, done=done)

    def __call__(self, prompt: str | list[ContentBlock], **kwargs) -> Any:
        """Enhanced call method that tracks trajectory."""
        # Start a new step with the user prompt as observation
        self._start_new_step(observation=prompt)

        # Let the original Strands Agent handle everything (including tool execution)
        result = super().__call__(prompt, **kwargs)

        # Finish the current step based on the result
        self._finish_current_step_from_result(result)

        return result

    async def invoke_async(self, prompt: str | list[ContentBlock], **kwargs) -> Any:
        """Async invoke method with trajectory tracking."""
        # Start a new step with the user prompt as observation
        self._start_new_step(observation=prompt)

        # Let the original Strands Agent handle everything
        result = None
        async for event in super().stream_async(prompt, **kwargs):
            if "result" in event:
                result = event["result"]
                break
        
        # Finish the current step based on the result
        if result:
            self._finish_current_step_from_result(result)
            
        return result


    def get_current_state(self) -> Step | None:
        """Get the current step state."""
        if self._trajectory.steps:
            return self._trajectory.steps[-1]
        return self._current_step

    def update_step_reward(self, reward: float):
        """Update the reward for the current or last step."""
        if self._current_step is not None:
            self._current_step.reward = reward
        elif self._trajectory.steps:
            self._trajectory.steps[-1].reward = reward
            # Update trajectory total reward
            self._trajectory.reward = sum(step.reward for step in self._trajectory.steps)

    def update_step_info(self, info: dict):
        """Update the info for the current or last step."""
        if self._current_step is not None:
            self._current_step.info.update(info)
        elif self._trajectory.steps:
            self._trajectory.steps[-1].info.update(info)
