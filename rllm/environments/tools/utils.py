import asyncio
import json

from openai import AsyncOpenAI

from rllm.environments.tools import ToolCaller

def chat_completion_with_tool(
    client: AsyncOpenAI,
    tool_caller: ToolCaller,
    messages_list,
    model="gpt-4",
    max_round=20,
    batch_size=32,  # Added batch_size parameter
):
    async def apply_tool(completion, messages, tool_caller, id=None):
        tool_calls = tool_caller.parse_tool_calls(completion.choices[0].message.content)

        if len(tool_calls) > 0:
            tool_call = tool_calls[0]
            if id is not None and isinstance(tool_call["parameters"], dict):
                tool_call["parameters"]["id"] = id
            tool_call_result = await tool_caller(tool_call["name"], tool_call["parameters"])
            print("tool_call_result", tool_call_result)
            messages.append(tool_call_result)
            return True

        return False

    async def tool_call_flow(example, request_id):
        try:
            messages = example["messages"]
            tool_infos = tool_caller.get_tool_infos()

            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_infos,
                temperature=0.6,
                max_tokens=8192,
                top_p=0.95,
                stop=["```\n\n"],
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": completion.choices[0].message.content + "```\n\n",
                }
            )
            print("round: 0", completion.choices[0].message.content)
            curr_round = 0
            while curr_round < max_round:
                use_tools = await apply_tool(
                    completion, messages, tool_caller, id=request_id
                )
                if use_tools:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tool_infos,
                        temperature=0.6,
                        max_tokens=8192,
                        top_p=0.95,
                        stop=["```\n\n"],
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": completion.choices[0].message.content
                            + "```\n\n",
                        }
                    )
                else:
                    break

                curr_round += 1
                print(f"round {curr_round}:", completion.choices[0].message.content)
        except Exception as e:
            print("Exception:", str(e))
            pass

        return example

    async def run_batch():
        # Initialize pool with first batch of requests
        active_requests = []
        results = []
        messages_iter = iter(messages_list)
        processed_count = 0
        request_id = 0

        # Fill initial pool
        for _ in range(batch_size):
            try:
                messages = next(messages_iter)
                task = asyncio.create_task(tool_call_flow(messages, request_id))
                active_requests.append((task, request_id))
                request_id += 1
            except StopIteration:
                break

        # Process requests and refill pool
        while active_requests:
            done, pending = await asyncio.wait(
                [task for task, _ in active_requests],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Update active_requests with pending tasks
            active_requests = [
                (task, id) for task, id in active_requests if task in pending
            ]

            for completed_task in done:
                # Find the ID for the completed task
                # task_id = next(id for task, id in active_requests if task == completed_task)
                final_messages = await completed_task
                # results.append({"id": task_id, "data": final_messages})
                results.append(final_messages)
                processed_count += 1

                # Save results checkpoint every 200 examples
                if processed_count % 600 == 0:
                    with open("messages_checkpoint.json", "w") as f:
                        json.dump(results, f, indent=2)

                # Try to add new request to maintain pool
                try:
                    messages = next(messages_iter)
                    new_task = asyncio.create_task(tool_call_flow(messages, request_id))
                    active_requests.append((new_task, request_id))
                    request_id += 1
                except StopIteration:
                    pass

            print("Active requests:", len(active_requests))

        return results

    return asyncio.run(run_batch())
