from vllm import EngineArgs
from vllm.engine.arg_utils import AsyncEngineArgs

from rllm.rollout.distributed import DistributedVLLM
from rllm.utils import save_jsonl

from vllm import SamplingParams

if __name__ == "__main__":
    model_name = "meta-llama/LLaMA-3.1.-8b-Instruct"

    engine_args = AsyncEngineArgs(
            model=model_name,
            device="cuda",
            tensor_parallel_size=2,
    )

    engine = DistributedVLLM(
        num_workers=1,
        engine_args=engine_args
    )

    engine.chat("hello", SamplingParams(), "id")

    engine.shutdown()