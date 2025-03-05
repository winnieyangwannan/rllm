
import httpx
import asyncio

REFERENCE_COUNT = 8
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5
GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"

# must enter secret key and engine id https://programmablesearchengine.google.com/controlpanel/all
subscription_key = ""
cx = ""

class GoogleSearch:
    """A tool for searching google."""
    client = None

    def __init__(self):
        self.name = "google_search"
        
    @property
    def info(self):
        return {
            "type": "function", 
            "function": {
                "name": self.name,
                "description": f"Search a query using the Google search engine, returning the top {REFERENCE_COUNT} results along with a short snippet about their contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "query to be submitted to Google search engine"
                        }
                    },
                    "required": ["query"]
                }
            }
        }


    async def execute(self, **kwargs):
        """Execute Python code in sandbox with given arguments."""
        if GoogleSearch.client == None:
            await self.init_client()
        contexts = await self.search_with_google(**kwargs)
        return "\n\n".join(
            [f"[url:[{c['link']}]] {c['snippet']}" for i, c in enumerate(contexts)]
        )

    
    async def init_client(self):
        GoogleSearch.client = httpx.AsyncClient()

    async def kill_client(self):
        await GoogleSearch.client.aclose()
        GoogleSearch.client = None

    async def search_with_google(self, query: str):
        """
        Search with google and return the contexts.
        """
        client = GoogleSearch.client
        params = {
            "key": subscription_key,
            "cx": cx,
            "q": query,
            "num": REFERENCE_COUNT,
        }
        
        response = await client.get(url = GOOGLE_SEARCH_ENDPOINT, params=params, timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT
        )
        if not response.is_success:
            print(f"{response.status_code} {response.text}")
        json_content = response.json()
        try:
            contexts = json_content["items"][:REFERENCE_COUNT]
        except KeyError:
            print(f"Error encountered: {json_content}")
            return []
        return contexts


if __name__ == '__main__':
    search = GoogleSearch()
    print(asyncio.run(search.execute(query = "How many words in Macbeth?")))