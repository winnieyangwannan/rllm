import asyncio
import os
import sys
import json
import traceback
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
from copy import deepcopy

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import AsyncOpenAI
from dotenv import load_dotenv
from rllm.tools.tool_base import Tool, ToolOutput
from rllm.agents.tool_agent import ToolAgent
from rllm.agents.system_prompts import TOOL_SYSTEM_PROMPT
from rllm.parser import get_tool_parser
from rllm.agents.agent import Trajectory
from rllm.data.dataset_types import TestDataset
from rllm.data.utils import load_dataset

load_dotenv()

class MCPTool(Tool):    
    def __init__(self, session, tool_name, tool_description, tool_schema):
        self._tool_schema = tool_schema
        self.session = session
        
        super().__init__(
            name=tool_name,
            description=tool_description
        )
        
    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._tool_schema
            }
        }
    
    async def async_forward(self, **kwargs) -> ToolOutput:
        try:
            print(f"Calling MCP tool: {self.name} with args: {kwargs}")
            
            result = await self.session.call_tool(self.name, kwargs)            
            if hasattr(result, 'content'):
                if hasattr(result.content, 'text'):
                    content_str = result.content.text
                elif isinstance(result.content, list) and hasattr(result.content[0], 'text'):
                    content_str = result.content[0].text
                else:
                    content_str = str(result.content)
            else:
                content_str = str(result)
                
            print(f"MCP tool result: {content_str}")
            return ToolOutput(name=self.name, output=content_str)
        except Exception as e:
            print(f"Error executing MCP tool {self.name}: {str(e)}")
            traceback.print_exc()
            return ToolOutput(name=self.name, error=f"Error: {str(e)}")

class MCPToolAgent(ToolAgent):
    def __init__(self, model_name="", parser_name="qwen", tools=[]):
        self.system_prompt = TOOL_SYSTEM_PROMPT
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        
        parser_class = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class(model=model_name)
        self.model_name = model_name  

        tools_json = [tool.json for tool in tools]
        self.tools_prompt = self.tool_parser.get_tool_prompt(
            json.dumps(tools_json, indent=2)
        )
        
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0
        self.reset()
    
    def get_tool_by_name(self, tool_name: str) -> Optional[Tool]:
        return self.tool_map.get(tool_name)

class MCPClient:
    def __init__(self, model_base_url="http://localhost:30000/v1", model_name="Qwen/Qwen3-4B"):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        self.client = AsyncOpenAI(
            base_url=model_base_url,
            api_key="EMPTY",
        )
        
        self.agent = None
        self.mcp_tools = []
        self.model_name = model_name
    
    async def connect_to_server(self, server_command: str, server_args: List[str] = None, env: Dict[str, str] = None):
        server_params = StdioServerParameters(
            command=server_command,
            args=server_args or [],
            env=env
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to MCP server with tools:", [tool.name for tool in tools])
        
        self.mcp_tools = []
        for tool in tools:
            mcp_tool = MCPTool(
                session=self.session,
                tool_name=tool.name,
                tool_description=tool.description,
                tool_schema=tool.inputSchema
            )
            self.mcp_tools.append(mcp_tool)
        
        self.agent = MCPToolAgent(
            model_name=self.model_name, 
            parser_name="qwen",  
            tools=self.mcp_tools
        )
        
        print(f"Agent initialized with {len(self.mcp_tools)} tools")
    
    async def process_query(self, query: str, max_rounds: int = 5) -> str:
        if not self.agent:
            return "Error: Agent not initialized. Please connect to an MCP server first."
        
        self.agent.reset()
        
        self.agent.update_from_env(
            observation=query,
            reward=0.0,
            done=False,
            info={}
        )
        
        done = False
        final_text = []
        current_round = 0
        
        while not done and current_round < max_rounds:
            messages = self.agent.chat_completions
            print(f"\nSending query to model (round {current_round + 1})...")
            try:
                response = await self.client.chat.completions.create(
                    model="qwen3",
                    messages=messages,
                    tools=[tool.json for tool in self.mcp_tools],
                    tool_choice="auto"
                )
                
                # Extract the response content properly
                assistant_message = response.choices[0].message
                content = assistant_message.content or ""
                tool_calls = assistant_message.tool_calls or []
                
                # Update agent with the OpenAI response - pass the raw content, not the response object
                self.agent.update_from_model(content)
                
                # Check if there's content to add to final text
                if content:
                    final_text.append(content)
                
                if tool_calls:
                    print(f"\n[Processing {len(tool_calls)} tool calls]")
                    
                    tool_outputs = {}
                    
                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        tool_args_str = tool_call.function.arguments
                        tool_id = tool_call.id
                        
                        try:
                            tool_args = json.loads(tool_args_str)
                        except json.JSONDecodeError:
                            print(f"Error parsing tool arguments: {tool_args_str}")
                            tool_args = {"error": "Invalid JSON arguments"}
                        
                        print(f"\n[Calling tool {tool_name} with args {tool_args}]")
                        
                        tool_instance = self.agent.get_tool_by_name(tool_name)
                        if tool_instance:
                            print(f"Executing tool {tool_name}")
                            result = await tool_instance.async_forward(**tool_args)
                            result_str = result.to_string() if result else "No result"
                            print(f"Tool execution result: {result_str[:100]}...")
                            tool_outputs[tool_id] = result_str
                        else:
                            print(f"Tool {tool_name} not found!")
                            tool_outputs[tool_id] = f"Error: Tool {tool_name} not found"
                    
                    # Now we need to manually add the tool call messages to the agent's message history
                    # First, add the assistant message with tool calls
                    assistant_message_dict = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in tool_calls
                        ]
                    }
                    
                    # Replace the last assistant message with the proper format
                    if self.agent.messages and self.agent.messages[-1]["role"] == "assistant":
                        self.agent.messages[-1] = assistant_message_dict
                    else:
                        self.agent.messages.append(assistant_message_dict)
                    
                    # Add tool response messages
                    for tool_call in tool_calls:
                        tool_response = {
                            "role": "tool",
                            "content": tool_outputs[tool_call.id],
                            "tool_call_id": tool_call.id
                        }
                        self.agent.messages.append(tool_response)
                else: 
                    done = True
                    
                current_round += 1
                    
            except Exception as e:
                print(f"Error in query processing: {str(e)}")
                traceback.print_exc()
                return f"Error: {str(e)}"
        
        return "\n".join(final_text)

    async def chat_loop(self):
        print("\nMCP Client Started!")
        print(f"Connected to model at: {self.client.base_url}")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print(f"\nResponse: {response}")

            except Exception as e:
                print(f"\nError: {str(e)}")
                traceback.print_exc()

    async def cleanup(self):
        await self.exit_stack.aclose()

def process_math_fn(example, idx):
    print(example)
    question = example.pop("problem")
    instruction = "Let's think step by step, put your final answer within \\\\boxed{}, and write python to evaluate math expressions if needed."
    question = f"{question} {instruction}"
    answer = example.pop("answer")

    task = {
        "ground_truth": answer,
        "question": question,
        "idx": idx,
        'data_source': 'math' 
    }
    return task

def load_data(n=1, dataset_enum=None):
    dataset = load_dataset(dataset_enum)
    data = []
    for idx, example in enumerate(dataset):
        if isinstance(dataset_enum, TestDataset.Math): 
            processed = process_math_fn(example, idx)
        else:
            print(f"Warning: Unsupported dataset type {type(dataset_enum)} in this context. Skipping.")
            continue 
        for i in range(n):
            data.append(deepcopy(processed))
    return data

async def run_aime_evaluation():
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        
        print("\n--- Loading AIME Dataset ---")
        tasks = load_data(n=1, dataset_enum=TestDataset.Math.AIME)
        print(f"Loaded {len(tasks)} problems from AIME dataset.")

        all_results = []
        for i, task in enumerate(tasks):
            print(f"\n--- Processing Problem {i + 1}/{len(tasks)} ---")
            question = task["question"]
            ground_truth = task["ground_truth"]
            
            print(f"Question: {question}")
            print(f"Ground Truth: {ground_truth}")
            
            response = await client.process_query(question)
            
            print(f"Agent Response: {response}") 
            
            all_results.append({
                "problem_idx": i + 1,
                "question": question,
                "ground_truth": ground_truth,
                "agent_response": response
            })

        print("\n--- All AIME Problems Processed ---")
        # with open("aime_mcp_results.json", "w") as f:
        #     json.dump(all_results, f, indent=2)
        # print("Results saved to aime_mcp_results.json")

    finally:
        await client.cleanup()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <path_to_mcp_server.py>")
        print("This will run AIME evaluation using the specified MCP server")
        sys.exit(1)

    await run_aime_evaluation()

if __name__ == "__main__":
    asyncio.run(main()) 