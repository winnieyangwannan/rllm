try:
    from camel.toolkits import ExcelToolkit
    from camel.toolkits.function_tool import FunctionTool
except ImportError as e:
    print(e)

from rllm.tools.camel_tools.camel_tool_base import CamelTool


class ExtractExcelContentCamel(CamelTool):
    def __init__(self):
        super().__init__(function_tool=FunctionTool(ExcelToolkit().extract_excel_content))


if __name__ == "__main__":
    # Test ExtractExcelContentCamel
    excel_file = "rllm-internal/rllm/data/train/web/gaia_files/edd4d4f2-1a58-45c4-b038-67337af4e029.xlsx"
    print("Testing ExtractExcelContentCamel with file:", excel_file)
    result = ExtractExcelContentCamel()(**{"document_path": excel_file})
    print("ExtractExcelContentCamel result:", result)
