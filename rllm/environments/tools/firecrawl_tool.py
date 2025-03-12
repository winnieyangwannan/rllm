
from firecrawl import FirecrawlApp
import asyncio
import time

FIRECRAWL_API = ""
TIMEOUT = 300

class Firecrawl:
    """A tool for extracting data from websites."""
    app = None

    def __init__(self):
        self.name = "firecrawl"
        
    @property
    def info(self):
        return {
            "type": "function", 
            "function": {
                "name": self.name,
                "description": f"Scrapes a given url, returning content as markdown along with any links.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "url to scrape content from"
                        }
                    },
                    "required": ["url"]
                }
            }
        }


    async def execute(self, **kwargs):
        """ run firecrawl job asychronously """
        if Firecrawl.app == None:
            await self.init_app()
        try:
            job = self.start_firecrawl_job(**kwargs)
        except Exception as e:
            return f"Firecrawl job could not start: {e}"
        
        if not job['success']:
            return "Firecrawl job failed to start"
        
        job_id = job['id']

        start_time = time.monotonic()
        while True:
            status = Firecrawl.app.check_batch_scrape_status(job_id)
            if status['completed']:
                break
            await asyncio.sleep(1)
            if time.monotonic()-start_time > TIMEOUT:
                return "Firecrawl request timed out"

        if status['success']:

            return "\n\n".join(
                [f"<source_{i}>\n[url]: {page['metadata']['url']}\n[markdown]: {page['markdown']}\n</source_{i}>" for i, page in enumerate(status['data'])]
            )
        else:
            return "Firecrawl request errored"


    
    async def init_app(self):
        # Firecrawl.app = FirecrawlApp(api_key=FIRECRAWL_API)
        Firecrawl.app = FirecrawlApp(api_url="http://0.0.0.0:3002")

    def start_firecrawl_job(self, url):
        """
        start a job with firecrawl async api and return job id
        """
        
        # crawl has many scrape options, potentially can let the agent choose
        return Firecrawl.app.async_batch_scrape_urls(
            [url], 
            params= {
                'formats': ['markdown', 'links'],
                'onlyMainContent' : True
            }
        )



if __name__ == '__main__':
    search = Firecrawl()
    print(asyncio.run(search.execute(url = "https://agentica-project.com/")))