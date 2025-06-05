#!/usr/bin/env python3
import asyncio
import inspect
import logging
import os
import sys
import subprocess
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Union, TypeVar

from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='mcp_server.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

logger.info("MCP Server Script Starting Up...")


from rllm.tools import TOOL_REGISTRY
from rllm.tools.tool_base import ToolOutput


mcp = FastMCP("rllm_tools")

def format_tool_output(tool_output: ToolOutput) -> str:
    """Format the tool output to a string."""
    if tool_output.error:
        return f"Error: {tool_output.error}"
    elif tool_output.output is None:
        return ""
    elif isinstance(tool_output.output, (list, dict)):
        import json
        return json.dumps(tool_output.output)
    else:
        return str(tool_output.output)

calculator = TOOL_REGISTRY['calculator']()
lcb_python_interpreter = TOOL_REGISTRY['python']() 
local_python_interpreter = TOOL_REGISTRY['local_python']()
google_search = TOOL_REGISTRY['google_search']()
firecrawl = TOOL_REGISTRY['firecrawl']()
tavily = TOOL_REGISTRY['tavily']()
# audio2text = TOOL_REGISTRY['audio2text']()
# ask_question_about_audio = TOOL_REGISTRY['ask_question_about_audio']()
# extract_excel_content = TOOL_REGISTRY['extract_excel_content']()
# image_to_text = TOOL_REGISTRY['image_to_text']()
# ask_question_about_image = TOOL_REGISTRY['ask_question_about_image']()
# search_wiki = TOOL_REGISTRY['search_wiki']()
# search_google = TOOL_REGISTRY['search_google']()
# ask_video = TOOL_REGISTRY['ask_video']()
# browse_url = TOOL_REGISTRY['browse_url']()
# execute_code = TOOL_REGISTRY['execute_code']()

@mcp.tool()
async def calculator_tool(expression: str) -> str:
    """Evaluate mathematical expressions safely.
    
    Args:
        expression: The mathematical expression to evaluate
    """
    result = await calculator(expression, use_async=True)
    return format_tool_output(result)

def execute_python_locally(code: str) -> str:
    """Execute Python code locally using subprocess."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(code)
            temp_filename = f.name

        result = subprocess.run(
            [sys.executable, temp_filename],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        try:
            os.unlink(temp_filename)
        except:
            pass
        
        output = result.stdout
        if result.returncode != 0:
            output += f"\nError: {result.stderr}"
        
        return output
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 10 seconds."
    except Exception as e:
        return f"Error executing Python code: {str(e)}"

@mcp.tool()
async def python_interpreter_tool(code: str) -> str:
    """Execute Python code safely, with LCB primary and local subprocess fallback on LCB timeout."""
    logger.info("[MCP_SERVER_DEBUG] python_interpreter_tool ENTERED (LCB with execute_python_locally fallback)")
    
    fallback_output_str = execute_python_locally(code) 
    logger.info(f"execute_python_locally fallback finished. Output (first 200): {fallback_output_str[:200]}...")
    return fallback_output_str


    # lcb_server_timeout = 15  # Server-side timeout for the LCB call
    # lcb_internal_tool_timeout = lcb_server_timeout - 1 # Let LCB try to timeout first
    # if lcb_internal_tool_timeout < 1: 
    #     lcb_internal_tool_timeout = 1 # Ensure positive timeout

    # try:
    #     logger.info(f"Attempting code execution with LCBPythonInterpreter (tool timeout: {lcb_internal_tool_timeout}s, server wait timeout: {lcb_server_timeout}s)")
    #     logger.debug(f"LCB CODE BLOCK:\\n{code}")
        
    #     lcb_task = lcb_python_interpreter(code, timeout=lcb_internal_tool_timeout, use_async=True)
    #     lcb_result: ToolOutput = await asyncio.wait_for(lcb_python_interpreter(code, timeout=lcb_internal_tool_timeout, use_async=True), timeout=5)
        
    #     logger.info(f"LCBPythonInterpreter finished within server timeout. Error: '{lcb_result.error}', Stderr (first 200): '{str(lcb_result.stderr)[:200]}', Output (first 100): '{str(lcb_result.output)[:100]}'")
    #     return format_tool_output(lcb_result)

    # except asyncio.TimeoutError:
    #     logger.warning(f"LCBPythonInterpreter call EXPLICITLY TIMED OUT by server after {lcb_server_timeout}s. Falling back to execute_python_locally.")
    #     # The lcb_task was cancelled by asyncio.wait_for. 
    #     # LCBPythonInterpreter's internal mechanisms (signals, subprocess management) should ideally handle cleanup of its own child process.
        
    #     logger.info("Attempting code execution with execute_python_locally as fallback.")
    #     logger.debug(f"Fallback CODE BLOCK:\\n{code}")
    #     # execute_python_locally returns a string directly and has its own internal 10s timeout
    #     fallback_output_str = execute_python_locally(code) 
    #     logger.info(f"execute_python_locally fallback finished. Output (first 200): {fallback_output_str[:200]}...")
    #     return fallback_output_str # Direct string return
        
    # except Exception as e:
    #     # This catches other unexpected errors during the LCB attempt or asyncio.wait_for itself (not TimeoutError)
    #     logger.error(f"Unexpected error during LCBPythonInterpreter execution attempt: {type(e).__name__} - {str(e)}")
    #     logger.error(traceback.format_exc())
    #     return f"Error: Unexpected server error during LCB execution: {str(e)}"

@mcp.tool()
async def google_search_tool(query: str) -> str:
    """Search Google for a query and return results.
    
    Args:
        query: The search query
    """
    result = await google_search(query=query, use_async=True)
    return format_tool_output(result)

@mcp.tool()
async def firecrawl_tool(url: str) -> str:
    """Extract content from a web page using Firecrawl.
    
    Args:
        url: The URL to extract content from
    """
    result = await firecrawl(url, use_async=True)
    return format_tool_output(result)

@mcp.tool()
async def tavily_tool(query: str, search_depth: str = "basic") -> str:
    """Search the web using Tavily search engine.
    
    Args:
        query: The search query
        search_depth: Search depth (basic or advanced)
    """
    result = await tavily(query=query, search_depth=search_depth, use_async=True)
    return format_tool_output(result)


if __name__ == "__main__":
    mcp.run(transport='stdio') 