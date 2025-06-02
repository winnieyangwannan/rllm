import asyncio
import logging
import os
import time

import requests
from browser_pilot.entrypoint.client import CloudClient
from transformers import AutoTokenizer

from rllm.agents import WebArenaAgent
from rllm.environments.browsergym.browsergym_cloud import BrowserGymCloud

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("env.log")],
    force=True,
)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.CRITICAL + 1)
logging.getLogger("urllib3").setLevel(logging.CRITICAL + 1)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Create the environment (no batch_size parameter)
    client = CloudClient(url="ws://localhost:9999/send_and_wait", max_concurrency=128)
    n_parallel_agents = 1

    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    envs = [BrowserGymCloud(client=client) for i in range(n_parallel_agents)]

    agents = [WebArenaAgent() for i in range(n_parallel_agents)]

    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "model": model_name,
    }

    from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine

    class Config:
        def __init__(self, **kwargs):
            self.agent = AgentConfig(**kwargs)

    class AgentConfig:
        def __init__(self):
            self.disable_thinking = False

    engine = AsyncAgentExecutionEngine(
        config=Config(),
        agents=agents,
        envs=envs,
        rollout_engine=None,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30001/v1",
            "api_key": "token-abc123",
        },
        max_response_length=10000,
        max_prompt_length=30048,
        max_steps=20,
        n_parallel_agents=n_parallel_agents,
    )

    save_dir = "/data/sijun/rllm/tests/agent/webarena/logs"

    for i in range(5, 10):
        if not os.path.exists(save_dir + f"_{i}"):
            os.makedirs(save_dir + f"_{i}", exist_ok=True)
        files = os.listdir(save_dir + f"_{i}")
        finished_tasks = set(
            [int(f.split(".")[0].split("_")[1]) for f in files if f.endswith(".json")]
        )

        # tasks_to_run_full = set(range(600, 650)) | set(range(681, 689))
        tasks_to_run_full = set(range(812))

        tasks = [
            i
            for i in tasks_to_run_full
            if i not in finished_tasks
        ]
        tasks = [
            {
                "env_id": f"browsergym_async/webarena.{i}",
                "env_kwargs": {},
                "timeout": 30000,
                "slow_mo": 1000,
            }
            for i in sorted(tasks)
        ]
        logger.info(f"Running {len(tasks)} tasks")
        # print(tasks)

        def reset_webarena_instance(timeout=300):
            response = requests.get("http://108.214.96.13:7565/reset")
            logger.info(f"Starting new WebArena instance... {response.text}")
            start_time = time.time()
            ready = False
            while time.time() - start_time < timeout:
                response = requests.get("http://108.214.96.13:7565/status")
                logger.info(response.text)
                if "Ready for duty!" in response.text:
                    ready = True
                    break
                elif "Failed to connect to WebArena" in response.text:
                    ready = False
                    break
                elif "Reset ongoing" in response.text:
                    time.sleep(5)
                else:
                    time.sleep(5)
                    logger.info(f"Unknown response from WebArena = {response.text}")

            if not ready:
                raise Exception("WebArena instance not ready")

            return ready

        # timeout = 300
        # is_ready = False
        # for trial in range(3):
        #     is_ready = reset_webarena_instance(timeout=timeout)
        #     if is_ready:
        #         break
        #     else:
        #         logger.info(f"WebArena instance not ready, retry {trial + 1}/3")
        #         timeout += 100

        try:
            results = asyncio.run(engine.execute_tasks(tasks))
        except Exception as e:
            logger.info(f"Error: {e}")
            continue
