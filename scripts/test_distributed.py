
from rllm.data.load_dataset import Datasets, load_dataset
from rllm.rollout.distributed_client import DistributedLLMClient
from rllm.system_prompts import COT_MATH_SYSTEM_PROMPT


async def main():
    dataset = load_dataset(Datasets.AIME)[:5]
    messages_batch = [
        [
            {"role": "system", "content": COT_MATH_SYSTEM_PROMPT},
            {"role": "user", "content": entry["problem"]},
        ]
        for entry in dataset
    ]

    endpoints = [
        {"url": "http://0.0.0.0:8001/v1/chat/completions", "weight": 1.0},
    ]
    
    
    # Example sampling parameters
    sampling_params = {
        "temperature": 0.7,
        "model":"Qwen/QwQ-32B-Preview",
    }
    
    try:
        client = await DistributedLLMClient.create_client(endpoints)
        results = await client.chat_batch(messages_batch, sampling_params)
        
        # Print results
        for i, result in enumerate(results):
            print(f"Result {i}: {result}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'client' in locals():
            await client.__aexit__(None, None, None)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())