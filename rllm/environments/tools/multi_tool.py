from typing import List

from rllm.environments.tools.tool_base import Tool
from rllm.environments.tools import TOOL_REGISTRY
from rllm.environments.tools.toolcall_parser import ToolcallParser

class MultiTool(Tool):
    def __init__(self, tools: List[str], parser_type="json"):
        # Check if all tools are in the registry
        assert all(tool in TOOL_REGISTRY for tool in tools), "All tools must be in the registry TOOL_REGISTRY"

        # Initialize the tool map
        self.tool_map = {tool: TOOL_REGISTRY[tool]() for tool in tools}
        self.parser = ToolcallParser(parser_type=parser_type)

    @property
    def json(self):
        return [tool.json for tool in self.tool_map.values()]
    
    def forward(self, tool_name: str, *args, **kwargs):
        assert tool_name in self.tool_map, f"Tool {tool_name} not found in tool map"
        tool = self.tool_map[tool_name]
        return {
            "role": "tool",
            "name": tool_name,
            "content": tool(*args,**kwargs)
        }
    
    def parse_tool_calls(self, text):
        return self.parser.parse(text)

if __name__ == "__main__":
    multi_tool = MultiTool(["calculator", "firecrawl"])
    print(multi_tool.json)
    print(multi_tool("calculator", "1 + 2*3"))
    print(multi_tool("firecrawl", "https://www.yahoo.com"))
    
    # Create an async examples
    import asyncio
    async def test_async():
        tasks = [
            multi_tool("calculator", "1 + 2*3", use_async=True),
            multi_tool("firecrawl", "https://www.google.com", use_async=True)
        ]
        results = await asyncio.gather(*tasks)
        print(results)

    asyncio.run(test_async())