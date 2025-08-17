import asyncio
import logging
import os

import openai

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from rllm.parser.chat_template import ChatTemplateParser
from rllm.parser.tool_parser import ToolParser


class OpenAIEngine(RolloutEngine):
    def __init__(self, model: str, tokenizer=None, api_retries: int = 3, base_url: str = "https://api.openai.com/v1", api_key: str = os.getenv("OPENAI_API_KEY"), sampling_params: dict | None = None, **kwargs):
        self.model = model
        self.api_retries = api_retries
        self.sampling_params = sampling_params or {}

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
            try:
                self.tool_parser = ToolParser.get_parser(self.tokenizer)
            except Exception:
                print(f"Warning: No tool parser found for {self.tokenizer.name_or_path}. Tool calls not be parsed.")
                self.tool_parser = None
            self._use_chat_completions = False
        else:
            print("No tokenizer provided, will use the chat completions endpoint. This is not recommended.")
            self._use_chat_completions = True

        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    async def chat_completion(self, messages: list[dict], **kwargs) -> ModelOutput:
        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        retries = self.api_retries
        while retries > 0:
            try:
                response = await self.client.chat.completions.create(model=self.model, messages=messages, timeout=3600, **sampling_params)
                text = response.choices[0].message.content
                if hasattr(response.choices[0].message, "reasoning") and isinstance(response.choices[0].message.reasoning, str):
                    text = f"{THOUGHT_DELIMITER_START}\n{response.choices[0].message.reasoning}\n{THOUGHT_DELIMITER_END}\n\n{text}"
                return ModelOutput(text=text, tool_calls=response.choices[0].message.tool_calls, finish_reason=response.choices[0].finish_reason, completion_tokens=response.usage.completion_tokens, prompt_tokens=response.usage.prompt_tokens)
            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    raise Exception("Rate limit reached and retries exhausted.") from None
                print("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise Exception(f"Error processing content after retries: {e}") from e
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)

    async def completion(self, prompt: str, **kwargs) -> ModelOutput:
        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        sampling_params.pop("model")
        retries = self.api_retries
        while retries > 0:
            try:
                response = await self.client.completions.create(model=self.model, prompt=prompt, timeout=3600, **sampling_params)
                return ModelOutput(text=response.choices[0].text, tool_calls=[], finish_reason=response.choices[0].finish_reason, completion_tokens=response.usage.completion_tokens, prompt_tokens=response.usage.prompt_tokens)
            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    raise Exception("Rate limit reached and retries exhausted.") from None
                print("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise Exception(f"Error processing content after retries: {e}") from e
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        kwargs.pop("application_id", None)  # only needed for verl engine
        if self._use_chat_completions:
            return await self.chat_completion(messages, **kwargs)
        else:
            prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
            output = await self.completion(prompt, **kwargs)
            if self.tool_parser is not None:
                output.tool_calls = self.tool_parser.parse(output.text)
            return output
