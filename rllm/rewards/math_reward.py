"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""

from enum import Enum

from rllm.rewards import Reward, RewardInput, RewardOutput, RewardType
from rllm.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from rllm.system_prompts import ORM_PROMPT
from rllm.utils import call_gemini_llm

ORM_PROMPT_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""


class AnswertoReward(Enum):
    """
    Enum class for representing different types of answers and their corresponding rewards.

    Attributes:
        CORRECT (int): Represents a correct answer, associated with a positive reward.
        INCORRECT (int): Represents an incorrect answer, associated with a negative reward.
        NO_ANSWER (int): Represents a situation where no answer was provided, associated with a negative reward.
        FORMAT_ERROR (int): Represents an answer that has a formatting error, associated with a negative reward.
        UNK (int): Represents an unknown answer type, associated with a negative reward.
    """
    CORRECT = 1
    INCORRECT = -1
    FORMAT_ERROR = -1
    UNK = -1


class RewardMathFn(Reward):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        import pdb; pdb.set_trace()
        problem = input.problem
        # Process the LLM response
        model_response = input.model_response
        model_answer = extract_answer(model_response)
        if model_answer is None:
            return RewardOutput(reward=AnswertoReward.FORMAT_ERROR, is_correct=False)

        # Process the ground truth
        ground_truth = input.metadata.get("answer", None)
        if "\\boxed" in ground_truth:
            ground_truth = extract_answer(ground_truth)
        if ground_truth is None:
            return RewardOutput(reward=AnswertoReward.UNK, is_correct=False)

        # Grade the answer using sympy and mathd heuristics.
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return RewardOutput(reward=AnswertoReward.CORRECT, is_correct=True)

        # If latex heuristics fail, use LLM as ORM to evaluate correctness.
        orm_response = call_gemini_llm(
            system_prompt=ORM_PROMPT,
            prompt=ORM_PROMPT_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
            temperature=0.0,
        )
        if "[[YES]]" in orm_response:
            return RewardOutput(reward=AnswertoReward.CORRECT, is_correct=True)
        elif "[[NO]]" in orm_response:
            return RewardOutput(reward=AnswertoReward.INCORRECT, is_correct=False)
        return RewardOutput(reward=AnswertoReward.UNK, is_correct=False)


if __name__ == "__main__":
    reward = RewardMathFn()
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="The answer is \\boxed{the function 24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", metadata={"answer": "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"})
    output = reward(input)
    print(output)