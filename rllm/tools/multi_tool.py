from typing import List, Dict, Union, Optional, Type

from rllm.tools import tool_registry
from rllm.tools.tool_base import Tool, ToolOutput


class MultiTool(Tool):
    def __init__(self, tools: Optional[List[str]] = None, tool_map: Optional[Dict[str, Type[Tool]]] = None):
        """
        Initialize MultiTool with either tool names or a tool_map directly.
        
        Args:
            tools: List of tool names to look up in the registry (legacy behavior)
            tool_map: Dictionary mapping tool names to Tool classes (new behavior)
        """
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")
        
        if tool_map is not None:
            # New behavior: use provided tool_map with tool classes
            self.tools = list(tool_map.keys())
            # Instantiate tool classes with the name parameter
            self.tool_map = {}
            for name, tool_cls in tool_map.items():
                self.tool_map[name] = tool_cls(name=name)
        elif tools is not None:
            # Legacy behavior: look up tools in registry
            assert all(tool in tool_registry for tool in tools), "All tools must be in the registry"
            self.tools = tools
            self.tool_map = {tool: tool_registry.instantiate(tool) for tool in tools}
        else:
            # Default to empty
            self.tools = []
            self.tool_map = {}

    @property
    def json(self):
        return [tool.json for tool in self.tool_map.values()]
    
    def forward(self, *args, tool_name: str, **kwargs) -> ToolOutput:
        if tool_name not in self.tool_map:
            return ToolOutput(name=tool_name, output=f"Tool {tool_name} not found in tool map")
        tool = self.tool_map[tool_name]
        return tool(*args,**kwargs)

if __name__ == "__main__":
    multi_tool = MultiTool(["calculator", "firecrawl"])
    print(multi_tool.json)
    print(multi_tool("1 + 2*3", tool_name="calculator"))
    print(multi_tool("https://www.yahoo.com", tool_name="firecrawl"))
    
    # Create an async examples
    import asyncio
    async def test_async():
        tasks = [
            multi_tool("1 + 2*3", tool_name="calculator", use_async=True),
            multi_tool("https://www.google.com", tool_name="firecrawl", use_async=True)
        ]
        results = await asyncio.gather(*tasks)
        print(results)

    asyncio.run(test_async())