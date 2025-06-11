## Dependencies

```bash
pip install mcp-cli
```

For Tavily demo, also install Node.js and the Tavily MCP server:
```bash
# Install Node.js (if not already installed)
# Then install Tavily MCP server
npx @tavily/mcp-server
```

## Files

- `mcp_client.py` - Core MCP client with refactored, reusable components
- `mcp_server.py` - Local MCP server exposing RLLM tools  
- `tavily_demo.py` - Interactive demo using Tavily MCP server for web search

## Setup

### Prerequisites

1. **Python environment** with RLLM installed
2. **Local model server** running on port 30000 (for inference)
3. **Node.js** (for Tavily demo)

### Environment Variables

Create a `.env` file in this directory with:

```bash
# For Tavily demo
TAVILY_API_KEY=tvly-your-api-key-here

# For RLLM tools (if using web search tools)
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-custom-search-engine-id
FIRECRAWL_API_KEY=your-firecrawl-key

## Usage

### 1. Tavily MCP Demo (Recommended)

The simplest way to try MCP with external search capabilities:

```bash
# Get Tavily API key from https://app.tavily.com/home
export TAVILY_API_KEY=tvly-your-key-here

# Run the demo
python tavily_demo.py
```

This connects to the external Tavily MCP server and provides an interactive chat interface for web search.

### 2. Local RLLM Tools Server

To use local RLLM tools via MCP:

```bash
# Terminal 1: Start the local MCP server
python mcp_server.py

# Terminal 2: Connect a client to it
python mcp_client.py mcp_server.py
```

This runs the AIME math evaluation using local tools like calculator and Python interpreter.

## Architecture

The refactored code provides these reusable components:

- **`MCPTool`** - Adapts MCP tools to RLLM's Tool interface
- **`MCPToolAgent`** - ToolAgent subclass that works with Tool instances
- **`MCPClient`** - Core client for connecting to any MCP server
- **`TavilyMCPDemo`** - Example usage with external MCP server
