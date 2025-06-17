try:
    from camel.toolkits import FileWriteToolkit
    from camel.toolkits.function_tool import FunctionTool
except ImportError as e:
    print(e)

from rllm.tools.camel_tools.camel_tool_base import CamelTool


class WriteToFileCamel(CamelTool):
    def __init__(self):
        super().__init__(function_tool=FunctionTool(FileWriteToolkit.write_to_file))


if __name__ == "__main__":
    # Test WriteToFileCamel
    content = "This is a test content\nWriting to file using WriteToFileCamel"
    filename = "test_output.txt"
    encoding = "utf-8"

    print("Testing WriteToFileCamel")
    print("Content:", content)
    print("Filename:", filename)
    print("Encoding:", encoding)
    try:
        result = WriteToFileCamel()(**{"content": content, "filename": filename, "encoding": encoding})
        print("WriteToFileCamel result:", result)
    except Exception as e:
        print("Error testing WriteToFileCamel:", str(e))
