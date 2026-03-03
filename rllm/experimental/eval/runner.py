"""EvalRunner: orchestrates parallel evaluation of a model on a dataset using an agent scaffold."""

from __future__ import annotations

import asyncio
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor

from tqdm.asyncio import tqdm_asyncio

from rllm.experimental.eval.results import EvalItem, EvalResult
from rllm.experimental.eval.types import AgentConfig, AgentFlow, Evaluator

logger = logging.getLogger(__name__)


class EvalRunner:
    """Orchestrates parallel evaluation using a two-stage pipeline.

    Stage 1: AgentFlow.run(task, config) -> Episode (trajectories without rewards)
    Stage 2: Evaluator.evaluate(task, episode) -> EvalOutput (reward + signals)

    The runner writes evaluation results back onto each trajectory and the episode.
    """

    def __init__(self, base_url: str, model: str, concurrency: int = 64, agent_metadata: dict | None = None):
        self.base_url = base_url
        self.model = model
        self.concurrency = concurrency
        self.agent_metadata = agent_metadata or {}
        self._executor = ThreadPoolExecutor(max_workers=concurrency)

    async def run(self, dataset, agent: AgentFlow, evaluator: Evaluator, agent_name: str = "") -> EvalResult:
        """Run evaluation on a dataset using the given agent and evaluator.

        Args:
            dataset: An iterable of task dicts (e.g., rllm.data.Dataset).
            agent: AgentFlow instance with a .run() method.
            evaluator: Evaluator instance with an .evaluate() method.
            agent_name: Name of the agent for reporting.

        Returns:
            EvalResult with per-example and aggregate metrics.
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def eval_one(idx: int, task: dict) -> EvalItem:
            async with semaphore:
                try:
                    config = AgentConfig(
                        base_url=self.base_url,
                        model=self.model,
                        session_uid=f"eval-{idx}",
                        metadata=dict(self.agent_metadata),
                    )

                    # Stage 1: Run agent flow (supports both sync and async agents)
                    if inspect.iscoroutinefunction(agent.run):
                        episode = await agent.run(task, config)
                    else:
                        loop = asyncio.get_event_loop()
                        episode = await loop.run_in_executor(self._executor, agent.run, task, config)

                    # Stage 2: Evaluate
                    eval_output = evaluator.evaluate(task, episode)

                    # Write back onto trajectories
                    for traj in episode.trajectories:
                        traj.reward = eval_output.reward
                        traj.signals = {s.name: s.value for s in eval_output.signals}

                    # Set episode-level correctness
                    episode.is_correct = eval_output.is_correct

                    return EvalItem(
                        idx=idx,
                        reward=eval_output.reward,
                        is_correct=eval_output.is_correct,
                        signals={s.name: s.value for s in eval_output.signals},
                    )
                except Exception as e:
                    logger.warning("Error evaluating example %d: %s", idx, e)
                    return EvalItem(idx=idx, reward=0.0, is_correct=False, error=str(e))

        task_coros = [eval_one(i, task) for i, task in enumerate(dataset)]
        items = await tqdm_asyncio.gather(*task_coros, desc="Evaluating")

        dataset_name = getattr(dataset, "name", "unknown") or "unknown"
        return EvalResult.from_items(dataset_name, self.model, agent_name, items)
