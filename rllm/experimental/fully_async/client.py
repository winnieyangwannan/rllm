import asyncio
from typing import Any

import httpx

from rllm.experimental.fully_async.protocol import OutputChunk, OutputWithVersion
from rllm.parser.tool_parser import ToolParser


class LLMCallError(Exception):
    """Raised by RolloutClient when an LLM call fails."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class RolloutClient:
    """Async LLM client supporting both token mode (training) and text mode (eval).

    Mode is auto-detected based on whether a tokenizer is provided:
    - Token mode (tokenizer provided): Uses /generate endpoint, returns token IDs + logprobs
    - Text mode (no tokenizer): Uses /v1/chat/completions endpoint, returns text only

    Both modes satisfy the LLMClient protocol used by MLEBenchAgent.
    """

    def __init__(
        self,
        router_url: str,
        tokenizer=None,
        max_concurrency: int = 4096,
        max_tokens=32768,
        model: str | None = None,  # Required for text mode
        api_key: str | None = None,  # Optional auth for the server
        timeout: float = 600.0,  # Timeout for text mode requests (seconds)
    ):
        self.router_url = router_url.rstrip("/")
        self.tokenizer = tokenizer
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._mode = "token" if tokenizer else "text"

        # Validate: model is required for text mode
        if self._mode == "text" and not self.model:
            raise ValueError("model is required when tokenizer is not provided (text mode)")

        # Only create parser in token mode (requires tokenizer)
        self.parser = ToolParser.get_parser(tokenizer) if tokenizer else None
        self._max_concurrency = max_concurrency

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self._max_concurrency,
                max_keepalive_connections=min(self._max_concurrency, 1000),
            ),
            timeout=httpx.Timeout(self.timeout if self._mode == "text" else None),
        )

        self.cur_version = 0
        self.max_tokens = max_tokens
        self.resume_event = asyncio.Event()
        self.resume_event.set()

    @property
    def mode(self) -> str:
        """Return current mode: 'text' or 'token'."""
        return self._mode

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    def set_version(self, version: int):
        self.cur_version = version

    async def _post(self, payload):
        # Block if paused - ensures no new requests after pause()
        await self.resume_event.wait()

        response = await self.client.post(self.router_url + "/generate", json=payload)
        response.raise_for_status()
        return response.json()

    def resume(self):
        self.resume_event.set()

    def pause(self):
        self.resume_event.clear()

    # ========== Low-Level API ==========

    async def generate(self, prompt_ids: list[int], sampling_params: dict) -> OutputWithVersion:
        """
        Generate with token IDs directly (low-level API).

        Args:
            prompt_ids: List of input token IDs
            sampling_params: SGLang sampling parameters dict

        Returns:
            OutputWithVersion with prompt_ids and output_chunks
        """
        output = OutputWithVersion(prompt_ids=prompt_ids, output_chunks=[])

        while True:
            # Block at start of each iteration
            await self.resume_event.wait()
            output, sampling_params = await self._generate(output, sampling_params)
            if output.finish_reason == "abort":
                continue
            else:
                return output

    async def _generate(self, output: OutputWithVersion, sampling_params: dict):
        """Internal generate that handles a single request/response cycle."""
        old_version = self.cur_version
        payload = {
            "input_ids": output.all_tokens(),
            "sampling_params": sampling_params,
            "return_logprob": True,
        }

        response = await self._post(payload)

        # finish_reason is a dict with "type" key, or None
        finish_reason_obj = response["meta_info"].get("finish_reason")
        output.finish_reason = finish_reason_obj["type"] if finish_reason_obj else "unknown"

        # output_token_logprobs is a list of tuples: [(log_prob, token_id, _), ...]
        output_token_logprobs = response["meta_info"].get("output_token_logprobs", [])
        # Ensure logprobs are Python floats (not tensors or nested structures)
        logprob_values = [float(log_prob) for log_prob, token_id, _ in output_token_logprobs]

        # TODO: delete this after testing
        output_ids = [token_id for _, token_id, _ in output_token_logprobs]
        assert output_ids == response["output_ids"], "output_ids mismatch, {} != {}".format(output_ids, response["output_ids"])

        chunk = OutputChunk(
            response_ids=response["output_ids"],
            response_logprobs=logprob_values,
            version=old_version if output.finish_reason == "abort" else self.cur_version,
        )
        output.append(chunk)

        # Adjust max_tokens for continuation
        max_tokens = sampling_params.get("max_new_tokens") or sampling_params.get("max_tokens")
        if max_tokens is None:
            return output, sampling_params

        sampling_params = sampling_params.copy()
        remaining = max_tokens - len(chunk.response_ids)
        if "max_new_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = remaining
        else:
            sampling_params["max_tokens"] = remaining

        return output, sampling_params

    # ========== High-Level Chat API ==========
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], OutputWithVersion]:
        """
        Generate chat completion and parse response into OpenAI message format.

        Args:
            messages: List of message dicts (OpenAI format)
            sampling_params: SGLang sampling params dict
            tools: List of tool definitions (OpenAI function calling format)

        Returns:
            (message_dict, output): Parsed message and raw OutputWithVersion
        """
        if self._mode == "text":
            return await self._chat_completion_text(messages, sampling_params, tools)
        else:
            return await self._chat_completion_token(messages, sampling_params, tools)

    async def _chat_completion_token(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], OutputWithVersion]:
        """Token mode: use /generate endpoint with tokenizer (existing logic)."""
        from rllm.experimental.fully_async.message_utils import parse_response

        if self.tokenizer is None:
            raise ValueError("tokenizer required for token mode chat_completion")

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)

        sampling_params = sampling_params or {}
        if sampling_params.get("max_new_tokens", None) is None:
            sampling_params["max_new_tokens"] = self.max_tokens - len(prompt_ids)

        output = await self.generate(prompt_ids, sampling_params)

        message = parse_response(self.tokenizer, self.parser, output.all_response_ids())
        return message, output

    async def _chat_completion_text(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], OutputWithVersion]:
        """Text mode: use /v1/chat/completions endpoint (OpenAI-compatible)."""
        params = sampling_params or {}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": params.get("temperature", 1.0),
        }
        if tools:
            payload["tools"] = tools
        if params.get("top_p") is not None:
            payload["top_p"] = params["top_p"]
        if params.get("max_new_tokens") is not None:
            payload["max_tokens"] = params["max_new_tokens"]
        elif params.get("max_tokens") is not None:
            payload["max_tokens"] = params["max_tokens"]

        await self.resume_event.wait()

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = await self.client.post(
                self.router_url + "/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise LLMCallError(f"Request timed out after {self.timeout}s", retryable=True) from e
        except httpx.HTTPStatusError as e:
            retryable = e.response.status_code in (429, 500, 502, 503, 504)
            raise LLMCallError(f"HTTP {e.response.status_code}: {e}", retryable=retryable) from e
        except httpx.RequestError as e:
            raise LLMCallError(f"Connection error: {e}", retryable=True) from e

        data = response.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})

        output = OutputWithVersion(
            prompt_ids=[],
            output_chunks=[],
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            _completion_token_count=usage.get("completion_tokens", 0),
            _input_context_size=usage.get("prompt_tokens", 0),
        )
        return msg, output

    async def close(self):
        await self.client.aclose()
