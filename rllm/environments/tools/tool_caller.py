class ToolCaller:
    def __init__(self, tools):
        self.tool_map = {}
        for tool in tools:
            self.tool_map[tool.name] = tool

    async def __call__(self, tool_name, parameters):
        tool = self._get_tool(tool_name)
        if tool is not None:
            exec_res = await tool.execute(**parameters)
        else:
            exec_res = f"Function {tool_name} does not exist. "

        tool_call_result = {"role": "tool", "name": tool_name, "content": exec_res}
        return tool_call_result

    def _get_tool(self, tool_name):
        if tool_name not in self.tool_map:
            return None
        else:
            return self.tool_map[tool_name]

    def get_tool_infos(self):
        return [tool.info for tool in self.tool_map.values()]
