
import httpx
import asyncio

TAVILY_EXTRACT_ENDPOINT = "https://api.tavily.com/extract"


# https://docs.tavily.com/api-reference/endpoint/extract#body-extract-depth
API_KEY = "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"

class WebExtract:
    """A tool for extracting data from websites."""
    client = None

    def __init__(self):
        self.name = "web_extract"
        
    @property
    def info(self):
        return {
            "type": "function", 
            "function": {
                "name": self.name,
                "description": f"Extract web page content from one or more specified URLs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {
                                "type" : "string"
                            },
                            "description": "array of urls to extract content from"
                        }
                    },
                    "required": ["urls"]
                }
            }
        }


    async def execute(self, **kwargs):
        """Execute Python code in sandbox with given arguments."""
        if WebExtract.client == None:
            await self.init_client()
        return await self.tavily_extract(**kwargs)
        

    
    async def init_client(self):
        WebExtract.client = httpx.AsyncClient()

    async def kill_client(self):
        await WebExtract.client.aclose()
        WebExtract.client = None

    async def tavily_extract(self, urls):
        """
        Search with google and return the contexts.
        """
        client = WebExtract.client
        params = {
            "urls": urls,
            "include_images": False,
            "extract_depth": "basic"
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        response = await client.post(url = TAVILY_EXTRACT_ENDPOINT, json=params, headers=headers)
        if not response.is_success:
            print(f"{response.status_code} {response.text}")
        results = response.json()['results']
        return "\n\n".join(
            [f"[url={res['url']}]: {res['raw_content']}" for res in results]
        )


if __name__ == '__main__':
    search = WebExtract()
    print(asyncio.run(search.execute(urls = ["https://agentica-project.com/", "https://michaelzhiluo.github.io/"])))