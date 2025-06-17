try:
    from camel.toolkits import CodeExecutionToolkit
    from camel.toolkits.function_tool import FunctionTool
except ImportError as e:
    print(e)

from rllm.tools.camel_tools.camel_tool_base import CamelTool


class ExecuteCodeCamel(CamelTool):
    def __init__(self):
        super().__init__(function_tool=FunctionTool(CodeExecutionToolkit().execute_code))


if __name__ == "__main__":
    # Test ExecuteCodeCamel with a simple math expression
    code = "print(1 + 2)"
    try:
        result = ExecuteCodeCamel()(**{"code": code})
        print("\nExecuteCodeCamel result:", result)
    except Exception as e:
        print("Error testing ExecuteCodeCamel:", str(e))
