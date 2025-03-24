import json
import asyncio
import re
import string
from collections import Counter
from openai import AsyncOpenAI

from rllm.environments.tools import PythonInterpreter, GoogleSearch, Firecrawl
from rllm.environments.tools.tool_caller import ToolCaller

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))



async def _apply_tool(completion, messages, tool_caller):
    tool_calls = completion.choices[0].message.tool_calls
    if tool_calls != None:
        messages.append(completion.choices[0].message)
        for tool_call in tool_calls:
            tool_call_result = await tool_caller(
                tool_call.function.name, json.loads(tool_call.function.arguments)
            )
            tool_call_result['tool_call_id'] = tool_call.id
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
        print("round: 0", completion.choices[0].message.content)
        curr_round = 0
        while curr_round < max_round:
            use_tools = await _apply_tool(completion, messages, tool_caller)
            if use_tools:
                completion = await client.chat.completions.create(
                    model=model, messages=messages, tools=tool_infos
                )
            else:
                break

            curr_round += 1
            print(f"round {curr_round}:", completion.choices[0].message.content)
        
        messages.append(completion.choices[0].message)
        return messages[-1].content

    async def run_batch():
        tasks = [tool_call_flow(messages) for messages in messages_list]
        result = await asyncio.gather(*tasks)
        return result

    return asyncio.run(run_batch())


if __name__ == "__main__":

    DATASET_FILEPATH = ""

    with open(DATASET_FILEPATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    data_count = 5


    messages = [
        [
            {"role": "system", "content": "You are a question answering assistant. Your final message should be a single word or phrase, only provide the answer."},
            {"role": "user", "content": data["question"]}] 
            for data in dataset[:data_count] 
    ]

    client = AsyncOpenAI(
        api_key="" 
    )
    tool_caller = ToolCaller(tools=[GoogleSearch(), Firecrawl()])

    results = chat_completion_with_tool(
        client, tool_caller, messages
    )
    
    scores = [(f1_score(result, dataset[i]['answer']), exact_match_score(result, dataset[i]['answer'])) for i, result in enumerate(results)]

    avg_f1_score = sum([score[0][0] for score in scores])/data_count
    avg_precision_score = sum([score[0][1] for score in scores])/data_count
    avg_recall_score = sum([score[0][2] for score in scores])/data_count
    avg_exact_match_score = sum(score[1] for score in scores)/data_count

    breakpoint()


