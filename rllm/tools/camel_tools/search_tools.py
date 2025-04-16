try:
    from camel.toolkits import SearchToolkit
    from camel.toolkits.function_tool import FunctionTool
except ImportError as e:
    print(e)

from rllm.tools.camel_tools.camel_tool_base import CamelTool

class SearchWikiCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(SearchToolkit().search_wiki)
        )

class SearchLinkupCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(SearchToolkit().search_linkup)
        )

class SearchGoogleCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(SearchToolkit().search_google)
        )

class SearchDuckduckgoCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(SearchToolkit().search_duckduckgo)
        )

class QueryWolframAlphaCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(SearchToolkit().query_wolfram_alpha)
        )

class TavilySearchCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(SearchToolkit().tavily_search)
        )

class SearchBraveCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(SearchToolkit().search_brave)
        )

class SearchBochaCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(SearchToolkit().search_bocha)
        )

class SearchBaiduCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(SearchToolkit().search_baidu)
        )

class SearchBingCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(SearchToolkit().search_bing)
        )

if __name__ == "__main__":
    # Test SearchWikiCamel
    print("Testing SearchWikiCamel")
    result = SearchWikiCamel()(**{"entity": "Python programming language"})
    print("SearchWikiCamel result:", result)

    # Test SearchGoogleCamel
    print("\nTesting SearchGoogleCamel")
    result = SearchGoogleCamel()(**{
        "query": "latest AI developments",
        "num_result_pages": 2
    })
    print("SearchGoogleCamel result:", result)