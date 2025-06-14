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

# MCP (Model Context Protocol) Integration with AsyncAgentExecutionEngine

This example demonstrates how to use MCP servers as tool interfaces with the AsyncAgentExecutionEngine, similar to the pattern used in `examples/math_tool/run_math_with_tool.py`.

## Overview

The refactored `mcp_client.py` now follows the standard RLLM pattern:
- Uses `AsyncAgentExecutionEngine` for parallel execution
- Uses `ToolAgent` for agent logic
- Uses `MCPEnvironment` (custom) for MCP tool integration
- Follows the same evaluation pipeline as other tool examples

## Key Components

### MCPEnvironment
- Extends `BaseEnv` to work with `AsyncAgentExecutionEngine`
- Manages MCP server connections and tool discovery
- Handles tool execution asynchronously
- Integrates with the reward system for evaluation

### MCPTool
- Implements the `Tool` interface for MCP tools
- Provides async execution of MCP tool calls
- Handles error cases and result formatting

## Usage

1. **Start your MCP server** (example provided):
   ```bash
   python examples/mcp/math_mcp_server.py
   ```

2. **Run the evaluation**:
   ```bash
   python examples/mcp/mcp_client.py examples/mcp/math_mcp_server.py
   ```

## Differences from Original

### Before (Original mcp_client.py)
- Custom `MCPClient` class managing everything
- Manual query processing loop
- Direct OpenAI API calls
- Sequential processing of problems

### After (Refactored)
- Uses `AsyncAgentExecutionEngine` for execution
- Standard `ToolAgent` for agent logic
- Custom `MCPEnvironment` for MCP integration
- Parallel processing of multiple problems
- Integrated with RLLM's evaluation pipeline

## Benefits of Refactoring

1. **Consistency**: Follows the same pattern as other RLLM examples
2. **Scalability**: Supports parallel execution of multiple tasks
3. **Integration**: Works with RLLM's reward system and evaluation metrics
4. **Maintainability**: Uses standard RLLM components instead of custom implementations
5. **Extensibility**: Easy to add new MCP servers or modify evaluation logic

## Example MCP Server

The included `math_mcp_server.py` provides:
- `python`: Execute Python code (similar to built-in python tool)
- `finish`: Mark task completion with final answer

You can create your own MCP servers following the MCP specification and use them with this client.

## Configuration

The client is configured similarly to other RLLM examples:
- `n_parallel_agents`: Number of parallel agent instances
- `model_name`: Model to use for inference
- `max_steps`: Maximum steps per task
- `reward_fn`: Function to evaluate task completion

Note: MCP connections may not be fully thread-safe, so `n_parallel_agents` is set lower than typical tool examples.
