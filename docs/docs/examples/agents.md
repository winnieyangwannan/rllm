# Agent Examples

This page provides practical examples of using rLLM agents in real-world scenarios. Each example includes complete code that you can run and modify.

## Mathematical Problem Solving with MathAgent

The MathAgent is designed for mathematical reasoning tasks with step-by-step solutions.

### Basic Math Problem

```python
from rllm.agents import MathAgent

# Initialize the math agent
agent = MathAgent(remove_thinking=False)

# Problem to solve
problem = {
    "question": "Find the derivative of f(x) = 3x² + 2x - 1"
}

# Reset agent for new problem
agent.reset()

# Environment provides the problem
agent.update_from_env(
    observation=problem,
    reward=0.0,
    done=False,
    info={}
)

# Get the prompt for the language model
messages = agent.chat_completions
print("Messages for model:")
for msg in messages:
    print(f"{msg['role']}: {msg['content']}")

# Simulate model response
model_response = """
Let me find the derivative step by step.

Given: f(x) = 3x² + 2x - 1

Using the power rule: d/dx(xⁿ) = n·xⁿ⁻¹

For each term:
- d/dx(3x²) = 3 · 2x¹ = 6x
- d/dx(2x) = 2 · 1x⁰ = 2  
- d/dx(-1) = 0

Therefore: f'(x) = 6x + 2

\\boxed{f'(x) = 6x + 2}
"""

# Update agent with model response
agent.update_from_model(model_response)

# Get the current state
current_state = agent.get_current_state()
print(f"\nAction taken: {current_state.action}")
print(f"Thought process: {current_state.thought}")
```

### Multi-step Math Problem with Feedback

```python
from rllm.agents import MathAgent

agent = MathAgent(remove_thinking=True)  # Remove thinking tags
agent.reset()

# Initial problem
problem = {"question": "Solve the equation: 2x + 5 = 13"}
agent.update_from_env(problem, 0.0, False, {})

# First attempt (with error)
response1 = """
<think>
Let me solve this equation:
2x + 5 = 13
2x = 13 + 5  # Error here - should subtract 5
2x = 18
x = 9
</think>

To solve 2x + 5 = 13:
2x = 13 + 5 = 18
x = 18/2 = 9

\\boxed{x = 9}
"""

agent.update_from_model(response1)

# Environment provides feedback (incorrect answer)
feedback = "Your previous answer may contain a mistake. Please review it carefully and answer again. Put your final answer within \\boxed{}."
agent.update_from_env(feedback, -1.0, False, {})

# Corrected response
response2 = """
Let me solve this more carefully:
2x + 5 = 13
2x = 13 - 5  # Subtract 5 from both sides
2x = 8
x = 8/2 = 4

\\boxed{x = 4}
"""

agent.update_from_model(response2)

# Final check - correct answer
agent.update_from_env("Correct!", 1.0, True, {})

print("Final trajectory:")
for i, step in enumerate(agent.trajectory.steps):
    print(f"Step {i+1}: {step.action[:50]}...")
    print(f"Reward: {step.reward}")
```

## Tool Usage with ToolAgent

The ToolAgent can use external tools to solve complex problems requiring calculations, web search, or data processing.

### Calculator and Web Search Example

```python
from rllm.agents import ToolAgent

# Define mock tools for this example
class MockCalculator:
    def calculate(self, expression):
        try:
            result = eval(expression)  # Note: Use safely in real implementation
            return f"Result: {result}"
        except:
            return "Error in calculation"

class MockWebSearch:
    def search(self, query):
        # Mock search results
        if "population" in query.lower():
            return "The current world population is approximately 8 billion people."
        return f"Search results for: {query}"

# Initialize agent with tools
agent = ToolAgent(
    parser_name="qwen",
    tools=[MockCalculator(), MockWebSearch()]
)

# Problem requiring multiple tools
problem = {
    "question": "What is the population density of Earth if the surface area is 510.1 million km²?"
}

agent.reset()
agent.update_from_env(problem, 0.0, False, {})

# Model response with tool calls
model_response = """
I need to find the population density of Earth. Let me search for the current population first.

<function_call>
search("current world population 2024")
</function_call>
"""

agent.update_from_model(model_response)

# Simulate tool execution results
tool_results = {
    "tool_call_id_1": "The current world population is approximately 8 billion people."
}

agent.update_from_env(
    observation={"tool_outputs": tool_results},
    reward=0.0,
    done=False,
    info={}
)

# Continue with calculation
model_response2 = """
Now I'll calculate the population density using the population (8 billion) and surface area (510.1 million km²).

<function_call>
calculate("8000000000 / 510100000")
</function_call>
"""

agent.update_from_model(model_response2)

print("Tool interaction complete!")
print(f"Final messages: {len(agent.chat_completions)}")
```

## Web Browsing with WebAgent

The WebAgent can interact with web pages, navigate, and extract information.

### Basic Web Navigation

```python
from rllm.agents import WebAgent

# Initialize web agent
agent = WebAgent()
agent.use_axtree = True
agent.use_html = False
agent.use_screenshot = False

# Mock web observation
web_observation = {
    "goal_object": [
        {"type": "text", "text": "Find the latest news about artificial intelligence"}
    ],
    "open_pages_urls": ["https://example-news.com"],
    "open_pages_titles": ["Example News Site"],
    "active_page_index": 0,
    "axtree_txt": """
    button "Search" <click>55</click>
    textbox "Search" <click>123</click> {placeholder="Enter search terms"}
    link "AI News" <click>200</click>
    link "Technology" <click>250</click>
    heading "Latest Headlines" 
    article "AI Breakthrough in Language Models"
    """,
    "pruned_html": "<html><body>...</body></html>"
}

agent.reset()
agent.update_from_env(web_observation, 0.0, False, {})

# Model response with web action
model_response = """
I can see the webpage has a search textbox and some AI-related links. Let me click on the "AI News" link to find recent artificial intelligence news.

Thought: I should navigate to the AI News section to find the latest information.
Action: ```Click [200]```
"""

agent.update_from_model(model_response)

current_state = agent.get_current_state()
print(f"Web action: {current_state.action}")
print(f"Reasoning: {current_state.thought}")
```

### Advanced Web Interaction with Screenshots

```python
import numpy as np
from rllm.agents import WebAgent

agent = WebAgent()
agent.use_screenshot = True
agent.use_axtree = True

# Mock observation with screenshot
mock_screenshot = np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8)

web_observation = {
    "goal_object": [
        {"type": "text", "text": "Book a flight from NYC to LAX"}
    ],
    "open_pages_urls": ["https://travel-booking.com"],
    "open_pages_titles": ["Travel Booking"],
    "active_page_index": 0,
    "axtree_txt": """
    textbox "From" <click>100</click> {value=""}
    textbox "To" <click>150</click> {value=""}
    button "Search Flights" <click>200</click>
    """,
    "screenshot": mock_screenshot
}

agent.reset()
agent.update_from_env(web_observation, 0.0, False, {})

# Multi-step interaction
responses = [
    """
    I need to book a flight from NYC to LAX. I can see the flight booking form with "From" and "To" textboxes. Let me start by filling in the departure city.
    
    Thought: I need to enter "NYC" in the From field first.
    Action: ```Type [100]; NYC```
    """,
    """
    Now I'll enter the destination city LAX in the To field.
    
    Thought: Next step is to fill in the destination.
    Action: ```Type [150]; LAX```
    """,
    """
    Both cities are entered. Now I'll search for available flights.
    
    Thought: Ready to search for flights between NYC and LAX.
    Action: ```Click [200]```
    """
]

for i, response in enumerate(responses):
    agent.update_from_model(response)
    print(f"Step {i+1} action: {agent.get_current_state().action}")
    
    # Simulate environment response (except for last step)
    if i < len(responses) - 1:
        agent.update_from_env(web_observation, 0.0, False, {})
```

## Code Generation with CompetitionCodingAgent

The CompetitionCodingAgent iteratively improves code based on test feedback.

### Algorithm Problem Solving

```python
from rllm.agents import CompetitionCodingAgent

agent = CompetitionCodingAgent(
    remove_thinking=False,
    max_tests=3,
    public_test_only=True
)

# Programming problem
problem = {
    "question": """
    Given an array of integers, find the maximum sum of a contiguous subarray.
    
    Example:
    Input: [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6 (subarray [4,-1,2,1])
    
    Constraints:
    - Array length: 1 ≤ n ≤ 10^5
    - Elements: -10^4 ≤ arr[i] ≤ 10^4
    """
}

agent.reset()
agent.update_from_env(problem, 0.0, False, {})

# First attempt - brute force (inefficient)
first_attempt = """
I'll solve this using Kadane's algorithm, but let me start with a brute force approach to ensure correctness.

```python
def maxSubArray(nums):
    max_sum = float('-inf')
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            current_sum = sum(nums[i:j+1])
            max_sum = max(max_sum, current_sum)
    return max_sum
```

This solution checks all possible subarrays and returns the maximum sum.
"""

agent.update_from_model(first_attempt)

# Test results (some fail due to time limit)
test_results = [
    {
        "input": "[-2,1,-3,4,-1,2,1,-5,4]",
        "expected": "6",
        "output": "6",
        "passed": True
    },
    {
        "input": "[1]",
        "expected": "1", 
        "output": "1",
        "passed": True
    },
    {
        "input": "[-1,-2,-3,-4]",
        "expected": "-1",
        "output": "-1",
        "passed": True
    }
]

# All tests pass but might be slow
agent.update_from_env(
    observation={"test_results": test_results},
    reward=0.5,  # Partial credit for correctness
    done=False,
    info={}
)

# Improved solution
optimized_solution = """
The tests pass but the solution might be too slow for large inputs. Let me implement Kadane's algorithm for O(n) time complexity.

```python
def maxSubArray(nums):
    max_sum = nums[0]
    current_sum = nums[0]
    
    for i in range(1, len(nums)):
        # Either extend the existing subarray or start a new one
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

This optimized solution uses dynamic programming and runs in O(n) time with O(1) space.
"""

agent.update_from_model(optimized_solution)

# Final evaluation
agent.update_from_env(
    observation={"test_results": test_results},  # All tests still pass
    reward=1.0,  # Full credit
    done=True,
    info={}
)

print("Coding trajectory:")
for i, step in enumerate(agent.trajectory.steps):
    print(f"Step {i+1} reward: {step.reward}")
    if step.action:
        print(f"Code snippet: {step.action[:100]}...")
```

## Software Engineering with SWEAgent

The SWEAgent can navigate repositories, read files, and make code changes to fix issues.

### GitHub Issue Resolution

```python
from rllm.agents import SWEAgent

# XML-based parsing mode
agent = SWEAgent(use_fn_calling=False, format_model_response=True)

# Simulate a GitHub issue
github_issue = """
Bug Report: Function `calculate_average` returns incorrect result

The function in `utils/math.py` is supposed to calculate the average of a list of numbers, 
but it's returning the sum instead of the average.

Steps to reproduce:
1. Call `calculate_average([1, 2, 3, 4, 5])`
2. Expected: 3.0
3. Actual: 15

Please fix this function.
"""

agent.reset()
agent.update_from_env(github_issue, 0.0, False, {"max_steps": 10})

# Agent's first response - exploring the repository
response1 = """
I need to investigate this bug in the `calculate_average` function. Let me start by exploring the repository structure and then examining the problematic file.

<function=file_editor>
<parameter=command>view</parameter>
<parameter=path>/testbed</parameter>
</function>
"""

agent.update_from_model(response1)

# Simulate file system response
fs_response1 = """
Directory listing for /testbed:
- utils/
  - math.py
  - __init__.py
- tests/
  - test_math.py
- README.md
- setup.py
"""

agent.update_from_env(fs_response1, 0.0, False, {"max_steps": 9})

# Look at the problematic file
response2 = """
Now let me examine the `utils/math.py` file to understand the current implementation.

<function=file_editor>
<parameter=command>view</parameter>
<parameter=path>/testbed/utils/math.py</parameter>
</function>
"""

agent.update_from_model(response2)

# File content
file_content = """
def calculate_average(numbers):
    '''Calculate the average of a list of numbers'''
    total = sum(numbers)
    return total  # BUG: Should divide by len(numbers)

def calculate_sum(numbers):
    '''Calculate the sum of a list of numbers'''
    return sum(numbers)
"""

agent.update_from_env(file_content, 0.0, False, {"max_steps": 8})

# Fix the bug
response3 = """
I found the bug! The function is returning the total sum but not dividing by the number of elements. Let me fix this:

<function=file_editor>
<parameter=command>str_replace</parameter>
<parameter=path>/testbed/utils/math.py</parameter>
<parameter=old_str>def calculate_average(numbers):
    '''Calculate the average of a list of numbers'''
    total = sum(numbers)
    return total  # BUG: Should divide by len(numbers)</parameter>
<parameter=new_str>def calculate_average(numbers):
    '''Calculate the average of a list of numbers'''
    if not numbers:  # Handle empty list
        return 0
    total = sum(numbers)
    return total / len(numbers)</parameter>
</function>
"""

agent.update_from_model(response3)

# Test the fix
agent.update_from_env("File successfully modified.", 0.0, False, {"max_steps": 7})

response4 = """
Let me test the fix to make sure it works correctly:

<function=execute_bash>
<parameter=cmd>cd /testbed && python3 -c "from utils.math import calculate_average; print('Test 1:', calculate_average([1,2,3,4,5])); print('Test 2:', calculate_average([10,20,30])); print('Test 3:', calculate_average([]))"</parameter>
</function>
"""

agent.update_from_model(response4)

test_output = """
Test 1: 3.0
Test 2: 20.0  
Test 3: 0
"""

agent.update_from_env(test_output, 1.0, True, {"max_steps": 0})

print("SWE trajectory:")
for i, step in enumerate(agent.trajectory.steps):
    print(f"Step {i+1}: {step.action[:60]}...")
```

## Custom Agent Implementation

Here's an example of creating a custom agent for a specific domain:

### Custom Chess Agent

```python
from rllm.agents.agent import BaseAgent, Step, Trajectory

class ChessAgent(BaseAgent):
    """Custom agent for playing chess with move validation."""
    
    def __init__(self, skill_level="intermediate"):
        super().__init__()
        self.skill_level = skill_level
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0
        self.board_state = None
        
    def update_from_env(self, observation, reward, done, info, **kwargs):
        # Update previous step if exists
        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info
        
        if done:
            return
            
        # Extract board state and format for model
        if isinstance(observation, dict):
            self.board_state = observation.get("board", "")
            legal_moves = observation.get("legal_moves", [])
            game_status = observation.get("status", "ongoing")
            
            formatted_obs = f"""
            Current board position:
            {self.board_state}
            
            Legal moves: {', '.join(legal_moves)}
            Game status: {game_status}
            
            Please choose your next move in standard chess notation (e.g., e4, Nf3, O-O).
            Consider the position carefully and explain your reasoning.
            """
        else:
            formatted_obs = str(observation)
        
        # Add system message if first interaction
        if not self.messages:
            system_msg = f"""
            You are a {self.skill_level} chess player. Analyze positions carefully,
            consider tactics and strategy, and choose strong moves.
            Always explain your reasoning before making a move.
            """
            self.messages.append({"role": "system", "content": system_msg})
        
        self.messages.append({"role": "user", "content": formatted_obs})
        
        # Create new step
        current_step = Step(observation=observation, step=self.step)
        self._trajectory.steps.append(current_step)
    
    def update_from_model(self, response, **kwargs):
        content = response if isinstance(response, str) else response.choices[0].message.content
        
        # Parse chess move from response
        move = self._extract_chess_move(content)
        reasoning = self._extract_reasoning(content)
        
        # Update current step
        current_step = self._trajectory.steps[-1]
        current_step.model_response = content
        current_step.action = move
        current_step.thought = reasoning
        
        self.messages.append({"role": "assistant", "content": content})
        self.step += 1
    
    def reset(self):
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0
        self.board_state = None
    
    def get_current_state(self):
        if not self._trajectory.steps:
            raise ValueError("No steps in trajectory")
        return self._trajectory.steps[-1]
    
    @property
    def chat_completions(self):
        return self.messages
    
    @property
    def trajectory(self):
        return self._trajectory
    
    def _extract_chess_move(self, content):
        """Extract chess move from model response."""
        import re
        # Look for chess moves in standard notation
        move_pattern = r'\b([a-h][1-8]|[KQRBN][a-h1-8]?x?[a-h][1-8]|O-O-?O?)\b'
        matches = re.findall(move_pattern, content)
        return matches[0] if matches else "No valid move found"
    
    def _extract_reasoning(self, content):
        """Extract reasoning from model response."""
        # Simple extraction - everything before the move
        lines = content.split('\n')
        reasoning_lines = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['move:', 'i play', 'my move']):
                break
            reasoning_lines.append(line)
        return '\n'.join(reasoning_lines).strip()

# Example usage
chess_agent = ChessAgent(skill_level="advanced")

# Simulate a chess game
chess_observation = {
    "board": """
    r.bqkb.r
    pppp.ppp
    ..n..n..
    ....p...
    ....P...
    .....N..
    PPPP.PPP
    RNBQKB.R
    """,
    "legal_moves": ["d3", "d4", "Nc3", "Bb5", "a3", "h3"],
    "status": "white_to_move"
}

chess_agent.reset()
chess_agent.update_from_env(chess_observation, 0.0, False, {})

model_response = """
Looking at this position, I can see that Black has developed knights to c6 and f6, 
and has pushed the e-pawn to e5. This looks like an Italian Game opening.

I should continue developing my pieces and control the center. The move d3 supports 
my e4 pawn and prepares to develop my light-squared bishop. Alternatively, Nc3 
develops the knight and supports the center.

I think d3 is the most solid choice here as it strengthens my pawn structure.

My move: d3
"""

chess_agent.update_from_model(model_response)

current_state = chess_agent.get_current_state()
print(f"Chess move: {current_state.action}")
print(f"Reasoning: {current_state.thought}")
```

## Integration with Environments

Here's how to integrate agents with rLLM environments:

### Environment-Agent Loop

```python
from rllm.agents import MathAgent
from rllm.envs import MathEnv  # Hypothetical math environment

def run_agent_environment_loop(agent, env, num_episodes=5):
    """Run agent-environment interaction loop."""
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1} ===")
        
        # Reset for new episode
        agent.reset()
        observation = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Agent processes observation
            agent.update_from_env(observation, total_reward, done, {})
            
            # Get model response (simulated here)
            messages = agent.chat_completions
            model_response = simulate_model_call(messages)
            
            # Agent processes model response
            agent.update_from_model(model_response)
            
            # Get action from agent
            current_state = agent.get_current_state()
            action = current_state.action
            
            # Environment processes action
            observation, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"Action: {action[:50]}...")
            print(f"Reward: {reward}")
        
        print(f"Episode {episode + 1} completed. Total reward: {total_reward}")
        
        # Access full trajectory
        trajectory = agent.trajectory
        print(f"Episode had {len(trajectory.steps)} steps")

def simulate_model_call(messages):
    """Simulate a language model call."""
    # This would be replaced with actual model inference
    return "Simulated model response with mathematical reasoning..."

# Example usage
agent = MathAgent()
# env = MathEnv()  # Would be actual environment
# run_agent_environment_loop(agent, env)
```

These examples demonstrate the flexibility and power of rLLM agents across different domains. Each agent type is optimized for specific tasks while following the consistent BaseAgent interface, making them easy to use and extend for your specific needs. 