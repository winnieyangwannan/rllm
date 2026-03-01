"""Built-in function-calling agent flow for BFCL benchmark."""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)


_PYTHON_TO_JSON_SCHEMA_TYPE = {
    "float": "number",
    "dict": "object",
    "tuple": "array",
}


def _clean_schema(obj):
    """Recursively clean a BFCL function schema for OpenAI compatibility.

    Fixes two issues in the BFCL dataset:
    1. Irrelevant properties set to None (removes them).
    2. Python type names (float, dict, tuple) instead of JSON Schema types.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if v is None:
                continue
            v = _clean_schema(v)
            if k == "type" and isinstance(v, str):
                v = _PYTHON_TO_JSON_SCHEMA_TYPE.get(v, v)
            cleaned[k] = v
        return cleaned
    if isinstance(obj, list):
        return [_clean_schema(item) for item in obj]
    return obj


BFCL_SYSTEM_PROMPT = """\
You are a helpful assistant with access to functions/tools. When the user asks you \
to perform a task, call the appropriate function with the correct arguments. \
Respond ONLY with function calls when tools are available."""


class BFCLAgentFlow:
    """Function-calling agent flow for BFCL benchmark.

    Sends tool definitions to the model and captures tool-call responses.
    Expects task to have "question" (messages) and "function" (tool schemas).
    """

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")

        # BFCL question is typically a list of message lists
        question_data = task.get("question", [])
        functions = task.get("function", [])

        # Build messages
        messages = [{"role": "system", "content": BFCL_SYSTEM_PROMPT}]
        if isinstance(question_data, list) and question_data:
            # BFCL format: list of conversation turns
            if isinstance(question_data[0], list):
                # Nested list format — take the first conversation
                for msg in question_data[0]:
                    messages.append(msg)
            elif isinstance(question_data[0], dict):
                messages.extend(question_data)
            else:
                messages.append({"role": "user", "content": str(question_data)})
        elif isinstance(question_data, str):
            messages.append({"role": "user", "content": question_data})

        # Build tools in OpenAI format
        tools = []
        for func in functions:
            params = _clean_schema(func.get("parameters") or {})
            tool = {
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": params,
                },
            }
            tools.append(tool)

        response_text = ""
        tool_calls_data = []
        try:
            kwargs = {
                "model": config.model,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools

            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            response_text = choice.message.content or ""

            # Extract tool calls if present
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls_data.append({
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    })
        except Exception as e:
            logger.warning("LLM call failed: %s", e)

        # Build answer representation
        answer = json.dumps(tool_calls_data) if tool_calls_data else response_text

        step = Step(
            input=json.dumps(messages[-1]) if messages else "",
            output=answer,
            done=True,
        )
        traj = Trajectory(name="solver", steps=[step])
        return Episode(
            task=task,
            trajectories=[traj],
            artifacts={
                "answer": answer,
                "tool_calls": tool_calls_data,
                "raw_response": response_text,
            },
        )


# Singleton instance for registry
bfcl_agent = BFCLAgentFlow()
