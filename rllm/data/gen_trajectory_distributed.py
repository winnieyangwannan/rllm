from vllm import SamplingParams

from rllm.data.load_dataset import Datasets, load_dataset

from rllm.rollout.distributed import DistributedVLLM
from rllm.system_prompts import COT_MATH_SYSTEM_PROMPT

if __name__ == "__main__":
    model = "Qwen/QwQ-32B-Preview"
    dataset = load_dataset(Datasets.AIME)[:20]
    messages = [
        [
            {"role": "system", "content": COT_MATH_SYSTEM_PROMPT},
            {"role": "user", "content": entry["problem"]},
        ]
        for entry in dataset
    ]

    engine = DistributedVLLM(num_workers=1, tensor_parallel_size=4, model=model)

    responses = engine.chat(messages, SamplingParams(temperature=1.0, max_tokens=32768))

    print(responses[0])
