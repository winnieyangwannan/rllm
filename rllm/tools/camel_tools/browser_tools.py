try:
    from camel.toolkits import BrowserToolkit
    from camel.toolkits.function_tool import FunctionTool
except ImportError as e:
    print(e)

from rllm.tools.camel_tools.camel_tool_base import CamelTool


class BrowseUrlCamel(CamelTool):
    def __init__(self):
        super().__init__(function_tool=FunctionTool(BrowserToolkit(headless=True).browse_url))


if __name__ == "__main__":
    # Test BrowseUrlCamel
    url = "https://en.wikipedia.org/wiki/Mind"
    task = "Summarize what you see on the page"
    round_limit = 3

    print("Testing BrowseUrlCamel")
    print("URL:", url)
    print("Task:", task)
    print("Round limit:", round_limit)
    try:
        result = BrowseUrlCamel()(**{"task_prompt": task, "start_url": url, "round_limit": round_limit})
        print("BrowseUrlCamel result:", result)
    except Exception as e:
        print("Error testing BrowseUrlCamel:", str(e))
