import asyncio
import os
import sys
from dotenv import load_dotenv

from mcp_client import MCPClient

load_dotenv()

class TavilyMCPDemo:    
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            print("Error: TAVILY_API_KEY environment variable not set")
            sys.exit(1)
        
        self.client = MCPClient(
            model_base_url="http://localhost:30000/v1",
            model_name="Qwen/Qwen3-4B"
        )
    
    async def start(self):
        try:
            print("\n📡 Connecting to Tavily MCP server...")
            await self.client.connect_to_server(
                server_command="npx",
                server_args=["-y", "tavily-mcp@0.1.3"],
                env={"TAVILY_API_KEY": self.api_key}
            )
            
            print(f"\n✅ Connected! Available tools:")
            for tool in self.client.mcp_tools:
                print(f"  • {tool.name}: {tool.description}")
            
            await self.run_interactive_demo()
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            await self.client.cleanup()

    async def run_interactive_demo(self):
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                print(f"\nProcessing: {query}")
                print("-" * 50)
                
                response = await self.client.process_query(query, max_rounds=3)
                
                print(f"\nResponse:")
                print(response)
                print("-" * 50)
                
            except Exception as e:
                print(f"\nError processing query: {str(e)}")

async def main():
    demo = TavilyMCPDemo()
    await demo.start()

if __name__ == "__main__":
    print("Starting Tavily MCP Demo...")    
    asyncio.run(main()) 