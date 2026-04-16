"""
Embedded vLLM client - runs vLLM directly in-process without HTTP server.

This client provides the same LLMClient interface as RolloutClient but uses
vLLM's AsyncLLM directly, avoiding HTTP overhead for single-machine eval.

Usage:
    from rllm.experimental.fully_async.embedded_vllm_client import EmbeddedVLLMClient

    client = await EmbeddedVLLMClient.create(
        model_path="/checkpoint/models/Qwen3.5-4B",
        tensor_parallel_size=1,  # number of GPUs
        max_model_len=32768,
    )

    message, output = await client.chat_completion(messages, tools=tools)
    await client.shutdown()

Note: This requires vLLM to be installed with GPU support.
"""

import uuid
from typing import Any

from rllm.experimental.fully_async.protocol import OutputChunk, OutputWithVersion


class EmbeddedVLLMClient:
    """Embedded vLLM client using AsyncLLM directly (no HTTP server).

    This provides the same interface as RolloutClient but runs vLLM in-process.
    Use this for single-machine eval to avoid HTTP overhead.
    """

    def __init__(
        self,
        engine,  # AsyncLLM instance
        tokenizer,
        max_tokens: int = 32768,
    ):
        """Private constructor - use create() factory method instead."""
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self._mode = "embedded"
        self.cur_version = 0

        # Import parser
        from rllm.parser.tool_parser import ToolParser

        self.parser = ToolParser.get_parser(tokenizer)

    @classmethod
    async def create(
        cls,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        enable_prefix_caching: bool = True,
        **kwargs,
    ) -> "EmbeddedVLLMClient":
        """Factory method to create an EmbeddedVLLMClient.

        Args:
            model_path: Path to HuggingFace model or model name
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            max_model_len: Maximum sequence length
            gpu_memory_utilization: Fraction of GPU memory to use
            dtype: Data type for model weights ("auto", "float16", "bfloat16")
            enable_prefix_caching: Enable KV cache prefix caching (RadixAttention)
            **kwargs: Additional vLLM AsyncEngineArgs

        Returns:
            Initialized EmbeddedVLLMClient ready for inference
        """
        # Lazy import vLLM to avoid import errors when not installed
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.usage.usage_lib import UsageContext
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError as e:
            raise ImportError("vLLM is required for EmbeddedVLLMClient. Install with: pip install vllm") from e

        # Load tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Create engine args
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            disable_log_requests=True,
            **kwargs,
        )

        # Create vLLM config and engine
        usage_context = UsageContext.LLM_CLASS
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)

        engine = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
        )

        # Reset caches
        await engine.reset_mm_cache()

        return cls(engine=engine, tokenizer=tokenizer, max_tokens=max_model_len)

    @property
    def mode(self) -> str:
        """Return current mode: 'embedded'."""
        return self._mode

    def set_version(self, version: int):
        """Set the version for tracking weight updates during training."""
        self.cur_version = version

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], OutputWithVersion]:
        """
        Generate chat completion using embedded vLLM engine.

        Args:
            messages: List of message dicts (OpenAI format)
            sampling_params: Sampling parameters (temperature, top_p, max_tokens, etc.)
            tools: List of tool definitions (OpenAI function calling format)

        Returns:
            (message_dict, output): Parsed message and OutputWithVersion
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        from rllm.experimental.fully_async.message_utils import parse_response

        # Apply chat template
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )

        # Build sampling params
        sampling_params = sampling_params or {}
        max_tokens = sampling_params.pop("max_tokens", None) or sampling_params.pop("max_new_tokens", 4096)

        vllm_sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=sampling_params.get("temperature", 0.7),
            top_p=sampling_params.get("top_p", 0.9),
            logprobs=0,  # Return logprobs
            **{k: v for k, v in sampling_params.items() if k not in ("temperature", "top_p")},
        )

        # Create prompt
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)

        # Generate
        request_id = str(uuid.uuid4())
        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=vllm_sampling_params,
            request_id=request_id,
        )

        # Collect output
        final_output = None
        async for output in generator:
            final_output = output

        if final_output is None:
            raise RuntimeError("vLLM generation returned no output")

        # Extract token IDs and logprobs
        output_ids = list(final_output.outputs[0].token_ids)
        logprobs = None
        if final_output.outputs[0].logprobs is not None:
            logprobs = [logprobs_dict[output_ids[i]].logprob for i, logprobs_dict in enumerate(final_output.outputs[0].logprobs)]

        # Determine finish reason
        finish_reason = final_output.outputs[0].finish_reason
        if finish_reason == "stop":
            finish_reason = "completed"
        elif finish_reason == "length":
            finish_reason = "completed"

        # Build OutputWithVersion
        output = OutputWithVersion(
            prompt_ids=prompt_ids,
            output_chunks=[
                OutputChunk(
                    response_ids=output_ids,
                    response_logprobs=logprobs or [],
                    version=self.cur_version,
                )
            ],
            finish_reason=finish_reason,
        )

        # Decode and parse response
        response_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        message_dict = parse_response(self.parser, response_text)

        return message_dict, output

    async def shutdown(self):
        """Shutdown the vLLM engine and release GPU resources."""
        if hasattr(self.engine, "shutdown"):
            await self.engine.shutdown()
        elif hasattr(self.engine, "abort_all_requests"):
            await self.engine.abort_all_requests()
