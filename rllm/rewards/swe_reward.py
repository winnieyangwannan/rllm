"""
This module contains the RewardSWEFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from rllm.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from rllm.system_prompts import CODE_ORM_PROMPT, CODE_PROGRESS_PROMPT
from rllm.utils import call_gemini_llm
from rllm.envs.swebench.run_eval import check_correctness
from rllm.globals import MODEL_NAME_OR_PATH

ORM_USER_TEMPLATE = """
Problem Statement: {problem}
Patch 1: {patch_1}
Patch 2: {patch_2}
"""

PROGRESS_USER_TEMPLATE = """
Problem Statement: {problem}
Action: {action}
"""

class RewardSWEFn(RewardFn):
    """
    Reward function for evaluating coding problems. 

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.CODE, \
            "Invalid problem type: expected 'CODE', but got '{}'".format(input.problem_type)

        problem = input.problem
        model_response = input.model_response
        instance_id = input.metadata.get("instance_id", None)
        ground_truth = input.metadata.get("patch", None)
        
        # Attempt to parse a patch from the model response
        patch_start = model_response.find("diff --git")

        if patch_start == -1:
            # No patch found, handle it as if there's no valid patch
            patch = ""
        else:
            patch_end = model_response.find("```", patch_start)
            if patch_end == -1:
                patch_end = len(model_response)
            patch = model_response[patch_start:patch_end].strip()
        
        if not patch:
            # Handle cases where there is no valid patch
            response = call_gemini_llm(
                system_prompt=CODE_PROGRESS_PROMPT,
                prompt=PROGRESS_USER_TEMPLATE.format(problem=problem, action=model_response),
                temperature=0.0,
            )
            if "[[ON TRACK]]" in response:
                return RewardOutput(reward=self.config.on_track_reward_reward, is_correct=False)
            elif "[[OFF TRACK]]" in response:
                return RewardOutput(reward=self.config.off_track_reward_reward, is_correct=False)
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Handle cases where a valid patch exists
        if self.config.use_code_orm:
            orm_response = call_gemini_llm(
                system_prompt=CODE_ORM_PROMPT,
                prompt=ORM_USER_TEMPLATE.format(problem=problem, patch_1=patch, patch_2=ground_truth),
                temperature=0.0,
            )
            if "[[YES]]" in orm_response:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)
            elif "[[NO]]" in orm_response:
                return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
 
        actions = {
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": MODEL_NAME_OR_PATH,
        }

        predictions = {instance_id: actions}
                
        metadata = input.metadata
        reward = check_correctness(
            instance_ids=metadata.get("instance_ids", ""),
            actions=predictions,
        )
        
        if reward > 0:
            return RewardOutput(reward=RewardConfig(correct_reward=reward), is_correct=True)
        else:
            return RewardOutput(reward=RewardConfig(incorrect_reward=reward), is_correct=False)
    

if __name__ == "__main__":
    reward = RewardSWEFn(RewardConfig)
    metadata = {
        "instance_id": "astropy__astropy-12907",
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "patch": """\
diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py
--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -242,7 +242,7 @@ def _cstack(left, right):
        cright = _coord_matrix(right, 'right', noutp)
    else:
        cright = np.zeros((noutp, right.shape[1]))
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right
 
     return np.hstack([cleft, cright])
    """,
    }
    model_response = """\
diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py
--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -242,7 +242,7 @@ def _cstack(left, right):
        cright = _coord_matrix(right, 'right', noutp)
    else:
        cright = np.zeros((noutp, right.shape[1]))
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right
 
     return np.hstack([cleft, cright])
    """
    input = RewardInput(
        problem="""
Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels
Consider the following model:

```python
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

cm = m.Linear1D(10) & m.Linear1D(5)
```

It's separability matrix as you might expect is a diagonal:

```python
>>> separability_matrix(cm)
array([[ True, False],
       [False,  True]])
```

If I make the model more complex:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
array([[ True,  True, False, False],
       [ True,  True, False, False],
       [False, False,  True, False],
       [False, False, False,  True]])
```

The output matrix is again, as expected, the outputs and inputs to the linear models are separable and independent of each other.

If however, I nest these compound models:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & cm)
array([[ True,  True, False, False],
       [ True,  True, False, False],
       [False, False,  True,  True],
       [False, False,  True,  True]])
```
Suddenly the inputs and outputs are no longer separable?

This feels like a bug to me, but I might be missing something?
""",
        problem_type=RewardType.CODE,
        model_response=model_response,
        metadata=metadata,
    )
    output = reward(input)
    print(output)