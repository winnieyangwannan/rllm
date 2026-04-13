# MLE-bench Tool Use Flow

## Full Tool Use Flow

When the agent makes a tool call, here's the complete flow:

```
1. LLM Response with tool_calls
   ↓
2. Parse tool_calls from response.choices[0].message.tool_calls
   ↓
3. For each tool_call:
   a. Extract: tool_call.id, tool_call.function.name, tool_call.function.arguments
   b. Parse arguments JSON → dict
   c. Execute via execute_tool(name, args, sandbox, ...)
   d. Get result string
   ↓
4. Append assistant message (with tool_calls) to messages
   ↓
5. Append tool result message:
   {
     "role": "tool",
     "tool_call_id": tool_call.id,
     "content": result_string
   }
   ↓
6. Next iteration: send updated messages to LLM
```

---

## Expected Message History Structure

```python
messages = [
    # Initial setup
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": INSTANCE_PROMPT},
    
    # Turn 1: LLM calls bash
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command": "ls /root/data"}'
                }
            }
        ]
    },
    # Tool result
    {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": "train.csv\ntest.csv\nsample_submission.csv"
    },
    
    # Turn 2: LLM calls create
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_def456",
                "type": "function",
                "function": {
                    "name": "create",
                    "arguments": '{"path": "/workspace/train.py", "content": "import pandas..."}'
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "call_def456",
        "content": "File created at /workspace/train.py"
    },
    
    # Turn 3: LLM calls bash to run training
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_ghi789",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command": "cd /workspace && python train.py"}'
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "call_ghi789",
        "content": "Training complete. Saved submission.csv"
    },
    
    # Turn 4: LLM calls submit (CSV mode)
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_jkl012",
                "type": "function",
                "function": {
                    "name": "submit",
                    "arguments": '{"train_path": "/workspace/train.py", "submission_path": "/workspace/submission.csv"}'
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "call_jkl012",
        "content": "<<SUBMISSION>>/workspace/submission.csv<</SUBMISSION>>"
    },
]
```

---

## Code Flow Summary

### Entry Point: `_run_agent_loop()`

```
_run_agent_loop(client, model, messages, sandbox, ...)
    │
    ├── tools = get_tools(submit_file, check_submission_validity)
    │
    └── while turn < max_turns and not terminal:
            │
            ├── response = client.chat.completions.create(
            │       model=model,
            │       messages=messages,
            │       tools=tools,
            │       tool_choice="required",
            │   )
            │
            ├── assistant_msg = response.choices[0].message
            │
            ├── if assistant_msg.tool_calls:
            │       │
            │       ├── messages.append(assistant_msg)  # Add assistant turn
            │       │
            │       └── for tool_call in assistant_msg.tool_calls:
            │               │
            │               ├── name = tool_call.function.name
            │               ├── args = json.loads(tool_call.function.arguments)
            │               │
            │               ├── result = execute_tool(name, args, sandbox, submit_file, ...)
            │               │
            │               ├── messages.append({
            │               │       "role": "tool",
            │               │       "tool_call_id": tool_call.id,
            │               │       "content": result,
            │               │   })
            │               │
            │               ├── steps.append(Step(...))
            │               │
            │               └── if "<<SUBMISSION>>" in result:
            │                       terminal = True
            │                       pred_solution = extract_path(result)
            │
            └── turn += 1
```

### Tool Execution: `execute_tool()`

```
execute_tool(name, args, sandbox, submit_file, session_timeout, task_id, mle_bench_data_dir)
    │
    ├── "bash" → sandbox.exec(args["command"], timeout=session_timeout)
    │             → truncate_output(stdout + stderr)
    │
    ├── "edit" → sandbox.exec(f"cat {path}")
    │             → apply search/replace
    │             → sandbox.exec(f"cat > {path}", stdin=new_content)
    │
    ├── "create" → sandbox.exec(f"cat > {path}", stdin=content)
    │
    ├── "submit" → (mode-dependent)
    │       │
    │       ├── submit_file="csv":
    │       │       execute_submit_csv(args, sandbox)
    │       │       → validate both train_path and submission_path exist
    │       │       → return "<<SUBMISSION>>{submission_path}<</SUBMISSION>>"
    │       │
    │       └── submit_file="code":
    │               execute_submit_code(args, sandbox)
    │               → validate path exists
    │               → return "<<SUBMISSION>>{path}<</SUBMISSION>>"
    │
    └── "check_submission_validity" → validate_submission(...)
            → load sample_submission.csv from mle_bench_data_dir
            → compare columns, dtypes, row count
            → return validation result
```

---

## Submit Modes

### CSV Mode (`submit_file="csv"`)

**Tool Definition:**
```python
SUBMIT_TOOL_CSV = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit your solution with training script and predictions CSV.",
        "parameters": {
            "type": "object",
            "properties": {
                "train_path": {
                    "type": "string",
                    "description": "Path to train.py that generates submission.csv"
                },
                "submission_path": {
                    "type": "string",
                    "description": "Path to submission.csv with predictions"
                },
            },
            "required": ["train_path", "submission_path"],
        },
    },
}
```

**Agent calls:**
```json
{"train_path": "/workspace/train.py", "submission_path": "/workspace/submission.csv"}
```

### Code Mode (`submit_file="code"`)

**Tool Definition:**
```python
SUBMIT_TOOL_CODE = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit your training script path. The evaluator will run it.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to training script (e.g., /workspace/train.py)"
                },
            },
            "required": ["path"],
        },
    },
}
```

**Agent calls:**
```json
{"path": "/workspace/train.py"}
```

---

## Available Tools

| Tool | Description | Args |
|------|-------------|------|
| `bash` | Execute shell command | `command: str` |
| `edit` | Search/replace in file | `path: str, old_str: str, new_str: str` |
| `create` | Create new file | `path: str, content: str` |
| `submit` | Submit solution | CSV: `train_path, submission_path` / Code: `path` |
| `check_submission_validity` | Validate CSV format | `path: str` |

---

## Terminal Condition

The agent loop terminates when:
1. `submit` tool returns `<<SUBMISSION>>path<</SUBMISSION>>`
2. `max_turns` reached
3. `rollout_timeout` exceeded
4. Unrecoverable error

The `pred_solution` extracted from the submission marker is passed to the evaluator.
