import asyncio
import json
import logging
from typing import Dict, List
import os
from pathlib import Path

from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.agents.agent import BaseAgent
from rllm.parser import get_tool_parser
from rllm.tools.multi_tool import MultiTool
from rllm.agents.tool_agent import ToolAgent

from rllm.data.dataset_types import TestDataset, TrainDataset
from rllm.data.utils import load_dataset
from copy import deepcopy
import re
import string

# Get the absolute path to the gaia_files directory, need to change for train vs test
script_dir = Path(__file__).parent
extra_files_path = str(script_dir / "rllm" / "data" / "train" / "web" / "gaia_files")  + "/"
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

def extract_answer(final_response):
    match = re.search(r"<answer>(.*?)</answer>", final_response)
    if match:
        return match.group(1)
    
    return ""

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


SYSTEM_PROMPT = """
===== RULES OF ASSISTANT =====
Never forget you are a helpful assistant to complete a task. You must leverage your available tools, try your best to solve the problem, and explain your solutions.

Please note that our overall task may be very complicated. Here are some tips that may help you solve the task:
<tips>
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.  
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
- When solving tasks that require web searches, check Wikipedia first before exploring other websites.  
- When trying to solve math problems, you can try to write python code and use sympy library to solve the problem.
- Always verify the accuracy of your final answers! Try cross-checking the answers by other ways. (e.g., screenshots, webpage analysis, etc.).  
- Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.  
- After writing codes, do not forget to run the code and get the result. If it encounters an error, try to debug it. Also, bear in mind that the code execution environment does not support interactive input.
- When a tool fails to run, or the code does not run correctly, never assume that it returns the correct result and continue to reason based on the assumption, because the assumed result cannot lead you to the correct answer. The right way is to think about the reason for the error and try again.
- Search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding sources rather than direct answers, as it always need to use other tools to further process the url, e.g. interact with the webpage, extract webpage content, etc. 
- For downloading files, you can either use the web browser simulation toolkit or write codes.
</tips>
"""

def load_data(n=1):
    dataset = load_dataset(TrainDataset.Web.GAIA)
    questions = []
    answers = []
    # First collect all examples
    for idx, example in enumerate(dataset):
        question, answer = process_fn(example, idx)
        for i in range(n):
            questions.append(deepcopy(question))
            answers.append(deepcopy(answer))
    return questions, answers

def process_fn(example, idx):
    question = example.pop("Question")
    instruction = "Let's think step by step, put your final answer within <answer></answer> tags, and write python to evaluate math expressions if needed."
    # instruction = "Let's think step by step, put your final answer within <answer></answer> tags"
    question = f"{SYSTEM_PROMPT}\n{question}\n{instruction}"

    if example['file_name'] != "":
        question+=f"\nRelevant files, if any, are listed here: {extra_files_path + example.pop('file_name')}"

    # data = {
    #     "messages": [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": question},
    #     ],
    #     "problem": question,
    #     "answer": example.pop("Final answer"),
    #     "idx": idx
    # }
    return {"question": question}, {"answer": example.pop("Final answer")}


if __name__ == "__main__":
    # Create the environment (no batch_size parameter)
    camel_tools = ['audio2text', 'ask_question_about_audio', 'extract_excel_content', 'image_to_text', 'ask_question_about_image', 'search_wiki', 'search_google', 'execute_code', 'extract_document_content']
    # tasks = [
    #     SYSTEM_PROMPT + "\nA paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?",
    # ]

    # tasks = [{"question": task} for task in tasks]

    n = 165
    tasks, ground_truth = load_data()
    batch_size = 10
    tasks = tasks[:n]
    ground_truth = ground_truth[:n]

    # Create the batch agent with the tool agent
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


    from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine

    sampling_params = {
        "model": "gpt-4o",
        "temperature": 0.6,
        "max_tokens": 8192,
        "top_p": 0.95,
        "tools": ToolEnvironment(tools=camel_tools).tools.json,
    }
    
    async_agent_execution_engine = AsyncAgentExecutionEngine(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        agent_args = {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "tools": camel_tools
        },
        env_args = {
            "tools": camel_tools
        },
        engine_name="openai", 
        tokenizer=tokenizer,  # Using transformers tokenizer
        rollout_engine=None,
        sampling_params = sampling_params,
        max_episodes=30,
        n_parallel_agents=batch_size,
        max_trajectory_length=128000,
        max_prompt_length=10000,
    )

    # Run the environment interaction
    res = asyncio.run(async_agent_execution_engine.execute_tasks(tasks))
    answers = [extract_answer(r[-1]["response"]) for r in res]
    exact_match = [exact_match_score(a, b['answer']) for a, b in zip(answers, ground_truth)]
    print("avg score:", sum(exact_match)/len(exact_match))
    breakpoint()