import json
import asyncio
from openai import AsyncOpenAI

from rllm.environments.tools import PythonInterpreter
from rllm.environments.tools.utils import parse_tool_calls


class ToolCaller:
    def __init__(self, tools):
        self.tool_map = {}
        for tool in tools:
            self.tool_map[tool.name] = tool

    async def __call__(self, tool_name, parameters):
        tool = self._get_tool(tool_name)
        if tool is not None:
            exec_res = await tool.execute(**parameters)
        else:
            exec_res = f"Function {tool_name} does not exist. "

        tool_call_result = {"role": "tool", "name": tool_name, "content": exec_res}
        return tool_call_result

    def _get_tool(self, tool_name):
        if tool_name not in self.tool_map:
            return None
        else:
            return self.tool_map[tool_name]

    def get_tool_infos(self):
        return [tool.info for tool in self.tool_map.values()]


async def _apply_tool(completion, messages, tool_caller):
    tool_calls = parse_tool_calls(completion.choices[0].message.content)

    if len(tool_calls) > 0:
        tool_call = tool_calls[0]
        tool_call_result = await tool_caller(
            tool_call["name"], tool_call["parameters"]
        )
        print("tool_call_result", tool_call_result)
        messages.append(tool_call_result)
        return True
    
    return False


def chat_completion_with_tool(
    client: AsyncOpenAI,
    tool_caller: ToolCaller,
    messages_list,
    model="gpt-4o",
    max_round=20,
):
    async def tool_call_flow(messages):
        tool_infos = tool_caller.get_tool_infos()

        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_infos,
        )
        messages.append(
            {
                "role": "assistant",
                "content": completion.choices[0].message.content,
            }
        )
        print("round: 0", completion.choices[0].message.content)
        curr_round = 0
        while curr_round < max_round:
            use_tools = await _apply_tool(completion, messages, tool_caller)
            if use_tools:
                completion = await client.chat.completions.create(
                    model=model, messages=messages, tools=tool_infos
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": completion.choices[0].message.content,
                    }
                )
            else:
                break

            curr_round += 1
            print(f"round {curr_round}:", completion.choices[0].message.content)

        with open("messages.json", "w+") as f:
            json.dump(messages, f, indent=2)

        return messages

    async def run_batch():
        tasks = [tool_call_flow(messages) for messages in messages_list]
        result = await asyncio.gather(*tasks)

        return result

    return asyncio.run(run_batch())


if __name__ == "__main__":
    from rllm.data.dataset_types import TrainDataset
    from rllm.data.utils import load_dataset

    dataset = load_dataset(TrainDataset.DEEPSCALER)

    idx = 1

    messages = [
        [
            {
                "role": "user",
                "content": dataset[idx]["problem"]
                + "Let's think step by step, put your final answer within \\boxed{}, and use tools to evaluate math expressions if needed.",
            },
        ]
    ]

    # This is for model served with vLLM.
    client = AsyncOpenAI(
        base_url="http://0.0.0.0:8080/v1",
        api_key="EMPTY",
    )

    interpreter = PythonInterpreter()
    tool_caller = ToolCaller(tools=[interpreter])

    chat_completion_with_tool(
        client, tool_caller, messages, model="deepscaler-toolcall"
    )

    print("answer:", dataset[idx]['answer'])