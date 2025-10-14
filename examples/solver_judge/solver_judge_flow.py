import asyncio
import re

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class SolverJudgeWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, n_solutions: int = 2, reward_function: RewardFunction = None, **kwargs):
        super().__init__(rollout_engine, **kwargs)

        self.n_solutions = n_solutions
        self.reward_function = reward_function or countdown_reward_fn

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        problem = task["question"]

        # 1. generate candidate solutions in parallel
        async def generate_solution() -> ModelOutput:
            messages = [{"role": "user", "content": f"{problem}. Output the final answer within <answer>...</answer>"}]
            output: ModelOutput = await self.rollout_engine.get_model_response(messages)
            return output

        tasks = [asyncio.create_task(generate_solution()) for _ in range(self.n_solutions)]

        solutions = []
        rewards = []
        for completed_task in asyncio.as_completed(tasks):
            output = await completed_task

            solution = self._parse_solver_response(output.content)
            solutions.append(solution)

            reward = self.reward_function(task, solution).reward
            rewards.append(reward)

            traj = Trajectory(name="solver")
            traj.steps.append(
                Step(
                    chat_completions=[{"role": "user", "content": f"{problem}. Output the final answer within <answer>...</answer>"}] + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=solution,
                    reward=reward,
                    model_response=output,
                )
            )

            self.commit(trajectory=traj)

        # 2. select the best candidate solution
        judge_msgs = [{"role": "user", "content": self._create_judge_prompt(problem, solutions)}]
        output: ModelOutput = await self.rollout_engine.get_model_response(judge_msgs)

        solution_index = self._parse_judge_response(output.content)

        if solution_index < 0 or solution_index > self.n_solutions:  # invalid answer
            reward = 0.0
        elif solution_index == 0:  # no correct solution found
            reward = 1.0 if all(reward == 0.0 for reward in rewards) else 0.0
        else:
            reward = rewards[solution_index - 1]

        traj = Trajectory(name="judge")
        traj.steps.append(
            Step(
                chat_completions=judge_msgs + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                thought=output.reasoning,
                action=solution_index,
                reward=reward,
                model_response=output,
            )
        )

        self.commit(trajectory=traj)

        raise TerminationEvent(TerminationReason.ENV_DONE)

    def _create_judge_prompt(self, problem: str, solutions: list[str]) -> str:
        """Create a prompt for the judge to evaluate solutions."""
        prompt = f"""You are an expert verifier. Given a countdown problem and multiple solution attempts, select a correct solution.
Problem:
{problem}
Solutions to evaluate:
"""
        for i, solution in enumerate(solutions, 1):
            prompt += f"\nSolution {i}:\n{solution}\n"

        prompt += """
A correct solution must satisfy the following criteria:
1. The solution uses only the given numbers.
2. Each number is used exactly once.
3. Only basic arithmetic operations (+, -, *, /) are used.
4. The calculation results in the target number.
5. The final answer is clearly marked within <answer>...</answer> tags.
Output the index of your selected solution within <answer>...</answer> tags, e.g., <answer>1</answer> for the first solution, <answer>2</answer> for the second solution, etc. If multiple solutions are correct, output the index of the first correct solution. If no solution is correct, output 0."""
        return prompt

    def _parse_solver_response(self, response: str) -> str:
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return f"<answer>{answer_match.group(1).strip()}</answer>"
        else:
            return "No solution found"

    def _parse_judge_response(self, response: str) -> int:
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            try:
                solution_index = int(answer_text)
                return solution_index
            except ValueError:
                return -1
        else:
            return -1

    def assign_episode_correctness(self, episode: Episode) -> None:
        solver_rewards = [traj.steps[0].reward for traj in episode.trajectories if traj.name == "solver"]

        try:
            judge_trajectory = next(traj for traj in episode.trajectories if traj.name == "judge")
            judge_reward = judge_trajectory.steps[0].reward
        except StopIteration:
            raise ValueError("No judge trajectory found in episode") from None

        if any(reward == 1.0 for reward in solver_rewards) and judge_reward == 1.0:
            episode.is_correct = True
        else:
            episode.is_correct = False
