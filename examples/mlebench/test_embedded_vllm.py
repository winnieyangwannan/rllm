#!/usr/bin/env python3
"""
Test script for embedded vLLM mode (no HTTP server required).

This script demonstrates using vLLM's AsyncLLM directly in-process,
similar to how RLSI/verl uses it in vllm_async_server.py.

Usage:
    # Basic test (just verifies model loads and generates)
    python test_embedded_vllm.py

    # With specific model
    python test_embedded_vllm.py --model /checkpoint/agentic-models/winnieyangwn/models/Qwen3.5-4B

    # With config file
    python test_embedded_vllm.py --config configs/vllm_test.yaml
"""

import argparse
import asyncio
import sys


async def test_basic_generation(model_path: str, tensor_parallel_size: int = 1):
    """Test basic text generation with embedded vLLM."""
    print(f"\n{'=' * 60}")
    print("Testing EmbeddedVLLMClient")
    print(f"Model: {model_path}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"{'=' * 60}\n")

    # Import embedded client
    from rllm.experimental.fully_async.embedded_vllm_client import EmbeddedVLLMClient

    print("[1/4] Creating embedded vLLM client (loading model)...")
    client = await EmbeddedVLLMClient.create(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=8192,  # Use smaller context for testing
        gpu_memory_utilization=0.8,
    )
    print("      ✓ Model loaded successfully")
    print(f"      Mode: {client.mode}")

    print("\n[2/4] Testing simple chat completion...")
    messages = [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]

    message_dict, output = await client.chat_completion(
        messages=messages,
        sampling_params={"temperature": 0.1, "max_tokens": 50},
    )

    print("      ✓ Got response")
    print(f"      Content: {message_dict.get('content', '')[:200]}")
    print(f"      Tokens generated: {len(output.response_ids)}")
    print(f"      Finish reason: {output.finish_reason}")

    print("\n[3/4] Testing tool calling...")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}}, "required": ["command"]},
            },
        }
    ]

    messages = [{"role": "system", "content": "You are a helpful assistant. Use tools when needed."}, {"role": "user", "content": "List the files in the current directory."}]

    message_dict, output = await client.chat_completion(
        messages=messages,
        tools=tools,
        sampling_params={"temperature": 0.1, "max_tokens": 200},
    )

    print("      ✓ Got response")
    if message_dict.get("tool_calls"):
        print(f"      Tool calls: {len(message_dict['tool_calls'])}")
        for tc in message_dict["tool_calls"][:2]:
            print(f"        - {tc['function']['name']}: {tc['function']['arguments'][:100]}")
    else:
        print(f"      Content: {message_dict.get('content', '')[:200]}")

    print("\n[4/4] Shutting down...")
    await client.shutdown()
    print("      ✓ Shutdown complete")

    print(f"\n{'=' * 60}")
    print("All tests passed! ✓")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Test embedded vLLM mode")
    parser.add_argument("--model", type=str, default="/checkpoint/agentic-models/winnieyangwn/models/Qwen3.5-4B", help="Path to model")
    parser.add_argument("--config", type=str, help="Path to config YAML file (overrides --model)")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="Number of GPUs for tensor parallelism")

    args = parser.parse_args()

    model_path = args.model
    tensor_parallel_size = args.tensor_parallel_size

    # Load from config if provided
    if args.config:
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(args.config)
            model_path = cfg.model.name
            tensor_parallel_size = getattr(cfg.model, "tensor_parallel_size", 1)
            print(f"Loaded config from {args.config}")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            print("Using command line args instead")

    # Run test
    try:
        asyncio.run(test_basic_generation(model_path, tensor_parallel_size))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
