#!/usr/bin/env python3
"""
A simple MCP server that provides math tools including Python code execution.
"""

import asyncio
import os
import subprocess
import sys
import tempfile

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

# Create the server instance
server = Server("math-tools")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="python",
            description="Execute Python code and return the result",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool execution."""
    
    if name == "python":
        return await execute_python(arguments.get("code", ""))
    elif name == "finish":
        response = arguments.get("response", "")
        return [types.TextContent(type="text", text=f"Task completed: {response}")]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def execute_python(code: str) -> list[types.TextContent]:
    """Execute Python code in a subprocess and return the result."""
    try:
        # Create a temporary file to write the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute the Python code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=tempfile.gettempdir()
            )
            
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            if result.returncode != 0:
                output += f"Return code: {result.returncode}\n"
            
            if not output:
                output = "Code executed successfully with no output."
                
            return [types.TextContent(type="text", text=output)]
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file)
            except OSError:
                pass
                
    except subprocess.TimeoutExpired:
        return [types.TextContent(type="text", text="Error: Code execution timed out after 30 seconds")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error executing Python code: {str(e)}")]

async def main():
    # Run the server using stdio
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="math-tools",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 