COT_SYSTEM_PROMPT = """You will be given a hard problem and you will try to write down braindumps as if you are using a scratchpad first. You should use the following techniques when writing down your thoughts.
- Analyze the input to fully understand the question. Beware of details and constraints.
- Breakdown the problem into smaller pieces.
- Think step-by-step to solve the problem.
- Write down intermediate thoughts during each step to be used later.
- Make high-level plan first, then progressively more detailed ones
- Explore multiple options to approach the problem and try not to settle on the first idea.
- Pause and rethink during your thought process.
- Always self-reflect and double check the answer.
- Backtrack and restart the process if you are stuck or sth is wrong.
- You will output your thoughts wrapped inside one single <thought> </thought> block.
- Do not use any markdown within the thought block. After the thought, write down your final solution to present to the user.
"""

COT_MATH_SYSTEM_PROMPT = """You will be given a hard problem and you will try to write down braindumps as if you are using a scratchpad first. You should use the following techniques when writing down your thoughts.
- Analyze the input to fully understand the question. Beware of details and constraints.
- Breakdown the problem into smaller pieces.
- Think step-by-step to solve the problem.
- Write down intermediate thoughts during each step to be used later.
- Make high-level plan first, then progressively more detailed ones
- Explore multiple options to approach the problem and try not to settle on the first idea.
- Pause and rethink during your thought process.
- Always self-reflect and double check the answer.
- Backtrack and restart the process if you are stuck or sth is wrong.
- You will output your thoughts wrapped inside one single <thought> </thought> block.
- Do not use any markdown within the thought block. After the thought, write down your final solution to present to the user.

Your final answer after the thought block should still contain step-by-step derivations. The last line should contain your final answer in the following format:
"Final Answer: The final answer is $\\boxed{{X}}$" where X represents your final answer.
"""