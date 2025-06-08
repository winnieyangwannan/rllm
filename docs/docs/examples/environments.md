# Environment Examples

This page provides practical examples of using rLLM environments in real-world scenarios. Each example includes complete code that you can run and modify.

## Single-Turn Question Answering with SingleTurnEnvironment

The SingleTurnEnvironment is perfect for simple question-answering tasks where only one response is needed.

### Basic Q&A Example

```python
from rllm.environments import SingleTurnEnvironment

# Create a math question environment
env = SingleTurnEnvironment(
    task={
        "question": "What is the derivative of x^2 + 3x + 1?",
        "ground_truth": "2x + 3"
    }
)

# Reset environment and get initial observation
observation, info = env.reset()
print(f"Question: {observation['question']}")

# Agent provides answer
answer = "The derivative is 2x + 3"
next_obs, reward, terminated, truncated, info = env.step(answer)

print(f"Reward: {reward}")
print(f"Episode complete: {terminated}")

# Clean up
env.close()
```

### Custom Reward Function

```python
from rllm.environments import SingleTurnEnvironment
from rllm.rewards.reward_fn import RewardFunction, RewardOutput

def custom_math_reward(task_info, action):
    """Custom reward function for math problems."""
    question = task_info.get("question", "")
    ground_truth = task_info.get("ground_truth", "")
    
    # Simple string matching with partial credit
    if ground_truth.lower() in action.lower():
        reward = 1.0
        metadata = {"exact_match": True}
    elif any(term in action.lower() for term in ["derivative", "dx", "differentiate"]):
        reward = 0.5  # Partial credit for showing understanding
        metadata = {"partial_match": True}
    else:
        reward = 0.0
        metadata = {"no_match": True}
    
    return RewardOutput(reward=reward, metadata=metadata)

# Use custom reward function
env = SingleTurnEnvironment(
    task={
        "question": "Find the derivative of x^3",
        "ground_truth": "3x^2"
    },
    reward_fn=custom_math_reward
)

obs, info = env.reset()
result = env.step("The derivative using power rule is 3x^2")
print(f"Custom reward: {result[1]}")  # Should give 1.0
```

## Multi-Tool Problem Solving with ToolEnvironment

The ToolEnvironment enables agents to use external tools to solve complex problems.

### Calculator and Web Search Example

```python
from rllm.environments import ToolEnvironment

# Create environment with multiple tools
env = ToolEnvironment(
    task={
        "question": "What is the area of a circle with radius equal to the population of Monaco divided by 1000?"
    },
    tools=["web_search", "calculator"],
    max_steps=10
)

# Start the task
obs, info = env.reset()
print(f"Task: {obs['question']}")

# Step 1: Search for Monaco's population
tool_calls = [{
    "id": "search_1",
    "function": {
        "name": "web_search",
        "arguments": '{"query": "population of Monaco 2024"}'
    }
}]

next_obs, reward, done, info = env.step(tool_calls)
print(f"Search result: {next_obs['tool_outputs']['search_1']}")

# Step 2: Calculate radius (assuming population is ~39,000)
tool_calls = [{
    "id": "calc_1", 
    "function": {
        "name": "calculator",
        "arguments": '{"expr": "39000 / 1000"}'
    }
}]

next_obs, reward, done, info = env.step(tool_calls)
radius = next_obs['tool_outputs']['calc_1']
print(f"Radius: {radius}")

# Step 3: Calculate area using π * r^2
tool_calls = [{
    "id": "calc_2",
    "function": {
        "name": "calculator", 
        "arguments": f'{{"expr": "3.14159 * {radius}**2"}}'
    }
}]

next_obs, reward, done, info = env.step(tool_calls)
area = next_obs['tool_outputs']['calc_2']
print(f"Area: {area}")

# Step 4: Finish with final answer
finish_action = [{
    "function": {
        "name": "finish",
        "arguments": f'{{"response": "The area of the circle is {area} square units."}}'
    }
}]

final_obs, final_reward, done, info = env.step(finish_action)
print(f"Final reward: {final_reward}")
print(f"Task completed: {done}")

env.close()
```

### Error Handling with Tools

```python
from rllm.environments import ToolEnvironment

def safe_tool_execution(env, max_retries=3):
    """Example of robust tool usage with error handling."""
    
    obs, info = env.reset()
    step_count = 0
    
    while step_count < env.max_steps:
        try:
            # Try to use calculator
            tool_calls = [{
                "id": f"calc_{step_count}",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expr": "2 + 2"}'
                }
            }]
            
            next_obs, reward, done, info = env.step(tool_calls)
            
            if done:
                break
                
            # Check for errors in tool output
            tool_output = next_obs.get('tool_outputs', {})
            if 'error' in str(tool_output).lower():
                print(f"Tool error detected: {tool_output}")
                # Could implement retry logic here
            
            step_count += 1
            
        except Exception as e:
            print(f"Exception during tool execution: {e}")
            break
    
    return step_count, done

# Usage
env = ToolEnvironment(
    task={"question": "Calculate 2 + 2"},
    tools=["calculator"],
    max_steps=5
)

steps_taken, completed = safe_tool_execution(env)
print(f"Completed in {steps_taken} steps: {completed}")
```

## Competitive Programming with CompetitionCodingEnv

The CompetitionCodingEnv is designed for iterative code development with test feedback.

### Code Generation with Feedback

```python
from rllm.environments import CompetitionCodingEnv

# Create coding environment
env = CompetitionCodingEnv(
    task={
        "question": "Write a function that returns the nth Fibonacci number",
        "ground_truth": {
            "test_cases": [
                {"input": 0, "expected": 0},
                {"input": 1, "expected": 1}, 
                {"input": 5, "expected": 5},
                {"input": 10, "expected": 55}
            ]
        }
    },
    max_turns=3,
    reward_bonus_coeff=0.5  # Reward improvement between attempts
)

# First attempt - naive recursive solution
obs, info = env.reset()
print(f"Problem: {obs['question']}")

solution_v1 = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

next_obs, reward_1, done, info = env.step(solution_v1)
print(f"First attempt reward: {reward_1}")
print(f"Test results: {info}")

if not done:
    # Second attempt - optimized with memoization
    solution_v2 = """
def fibonacci(n):
    memo = {}
    def fib_helper(n):
        if n in memo:
            return memo[n]
        if n <= 1:
            memo[n] = n
        else:
            memo[n] = fib_helper(n-1) + fib_helper(n-2)
        return memo[n]
    return fib_helper(n)
"""
    
    next_obs, reward_2, done, info = env.step(solution_v2)
    print(f"Second attempt reward: {reward_2}")
    print(f"Improvement bonus applied: {reward_2 - reward_1}")

env.close()
```

### Handling Test Case Failures

```python
from rllm.environments import CompetitionCodingEnv

def iterative_code_development():
    """Example of improving code based on test feedback."""
    
    env = CompetitionCodingEnv(
        task={
            "question": "Implement binary search",
            "ground_truth": {
                "test_cases": [
                    {"input": [[1, 2, 3, 4, 5], 3], "expected": 2},
                    {"input": [[1, 3, 5, 7, 9], 7], "expected": 3},
                    {"input": [[2, 4, 6, 8], 5], "expected": -1}
                ]
            }
        },
        max_turns=3
    )
    
    obs, info = env.reset()
    
    # Attempt 1: Incorrect implementation
    buggy_code = """
def binary_search(arr, target):
    left, right = 0, len(arr)  # Bug: should be len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
    
    _, reward, done, info = env.step(buggy_code)
    print(f"Buggy code reward: {reward}")
    
    if not done and reward < 1.0:
        # Attempt 2: Fixed implementation
        fixed_code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1  # Fixed the bug
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
        
        _, reward, done, info = env.step(fixed_code)
        print(f"Fixed code reward: {reward}")
    
    env.close()
    return reward

final_score = iterative_code_development()
print(f"Final score: {final_score}")
```

## Repository Navigation with SWEEnv

The SWEEnv provides a realistic software development environment for fixing bugs and implementing features.

### File Exploration and Bug Fixing

```python
from rllm.environments import SWEEnv

# Create SWE environment with a specific issue
env = SWEEnv(
    entry={
        "problem_statement": "Fix the bug in the calculate_average function that causes division by zero",
        "repo": "example-python-project"
    },
    backend="docker",  # Use Docker for local development
    verbose=True
)

# Start the debugging session
obs, info = env.reset()
print(f"Issue: {obs}")

# Step 1: Explore the repository structure
action = "<function=execute_bash><parameter=command>find . -name '*.py' | head -10</parameter></function>"
obs, reward, done, info = env.step(action)
print(f"Python files found:\n{obs}")

# Step 2: Search for the problematic function
action = "<function=search><parameter=term>calculate_average</parameter></function>"
obs, reward, done, info = env.step(action)
print(f"Search results:\n{obs}")

# Step 3: View the problematic file
action = "<function=file_editor><parameter=command>view</parameter><parameter=path>utils.py</parameter></function>"
obs, reward, done, info = env.step(action)
print(f"File content:\n{obs}")

# Step 4: Fix the bug by adding zero division check
fixed_code = """
def calculate_average(numbers):
    if not numbers or len(numbers) == 0:
        return 0  # Handle empty list case
    return sum(numbers) / len(numbers)
"""

action = f"<function=file_editor><parameter=command>str_replace</parameter><parameter=path>utils.py</parameter><parameter=old_str>def calculate_average(numbers):\n    return sum(numbers) / len(numbers)</parameter><parameter=new_str>{fixed_code.strip()}</parameter></function>"

obs, reward, done, info = env.step(action)
print(f"Fix applied:\n{obs}")

# Step 5: Run tests to verify the fix
action = "<function=execute_bash><parameter=command>python -m pytest test_utils.py -v</parameter></function>"
obs, reward, done, info = env.step(action)
print(f"Test results:\n{obs}")

# Step 6: Finish the task
action = "<function=finish></function>"
obs, final_reward, done, info = env.step(action)

print(f"Task completed with reward: {final_reward}")
env.close()
```

### Implementation of New Features

```python
from rllm.environments import SWEEnv

def implement_feature():
    """Example of implementing a new feature in a codebase."""
    
    env = SWEEnv(
        entry={
            "problem_statement": "Add a new function 'is_palindrome' to the string utilities module",
        },
        step_timeout=120,  # Allow more time for complex operations
        verbose=False
    )
    
    obs, info = env.reset()
    
    # Explore the codebase
    steps = [
        "<function=execute_bash><parameter=command>ls -la</parameter></function>",
        "<function=search><parameter=term>string</parameter><parameter=file_pattern>*.py</parameter></function>",
        "<function=file_editor><parameter=command>view</parameter><parameter=path>string_utils.py</parameter></function>",
    ]
    
    for step in steps:
        obs, reward, done, info = env.step(step)
        print(f"Step output: {obs[:200]}...")  # Truncate for readability
        if done:
            break
    
    # Implement the palindrome function
    new_function = """
def is_palindrome(s):
    '''
    Check if a string is a palindrome.
    
    Args:
        s (str): Input string
        
    Returns:
        bool: True if palindrome, False otherwise
    '''
    # Remove spaces and convert to lowercase
    cleaned = ''.join(s.split()).lower()
    return cleaned == cleaned[::-1]
"""
    
    action = f"<function=file_editor><parameter=command>str_replace</parameter><parameter=path>string_utils.py</parameter><parameter=old_str># Add new functions here</parameter><parameter=new_str>{new_function}\n\n# Add new functions here</parameter></function>"
    
    obs, reward, done, info = env.step(action)
    print(f"Function added: {obs}")
    
    # Add tests for the new function
    test_code = '''
def test_is_palindrome():
    assert is_palindrome("racecar") == True
    assert is_palindrome("hello") == False
    assert is_palindrome("A man a plan a canal Panama") == True
    assert is_palindrome("") == True
'''
    
    action = f"<function=file_editor><parameter=command>create</parameter><parameter=path>test_palindrome.py</parameter><parameter=file_text>{test_code}</parameter></function>"
    
    obs, reward, done, info = env.step(action)
    
    # Run the tests
    action = "<function=execute_bash><parameter=command>python -m pytest test_palindrome.py -v</parameter></function>"
    obs, reward, done, info = env.step(action)
    
    # Finish the implementation
    action = "<function=finish></function>"
    obs, final_reward, done, info = env.step(action)
    
    env.close()
    return final_reward

score = implement_feature()
print(f"Implementation completed with score: {score}")
```

## Grid Navigation with FrozenLakeEnv

The FrozenLakeEnv demonstrates LLM training on spatial reasoning tasks.

### Basic Navigation

```python
from rllm.environments import FrozenLakeEnv

# Create a small grid for demonstration
env = FrozenLakeEnv(
    size=4,
    p=0.8,  # 80% frozen tiles, 20% holes
    is_slippery=False,  # Deterministic movement
    max_steps=15,
    seed=42
)

obs, info = env.reset()
print("Initial grid:")
print(env.render())

# Simple pathfinding strategy
actions = ["Right", "Right", "Down", "Down", "Right"]
total_reward = 0

for i, action in enumerate(actions):
    next_obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    print(f"\nStep {i+1}: Action '{action}'")
    print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    print("Current grid:")
    print(env.render())
    
    if terminated or truncated:
        print(f"Episode ended! Total reward: {total_reward}")
        break

env.close()
```

### LLM-Style Navigation with Text Descriptions

```python
from rllm.environments import FrozenLakeEnv

def text_based_navigation():
    """Example of using text-based observations for LLM training."""
    
    env = FrozenLakeEnv(size=4, seed=123, max_steps=20)
    obs, info = env.reset()
    
    # Describe the environment in natural language
    def describe_state():
        grid_text = env.render()
        description = f"""
        Current grid state:
        {grid_text}
        
        Legend:
        P = Player position
        _ = Frozen tile (safe)
        O = Hole (danger)
        G = Goal
        
        Player can move: Left, Down, Right, Up
        Goal: Reach G while avoiding O tiles.
        """
        return description
    
    print(describe_state())
    
    # Strategy: Try to find a safe path
    strategy_actions = ["Down", "Right", "Right", "Up", "Right"]
    
    for action in strategy_actions:
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nAction taken: {action}")
        print(f"Result: Reward={reward}, Done={terminated or truncated}")
        
        if terminated:
            if reward > 0:
                print("🎉 Successfully reached the goal!")
            else:
                print("💀 Fell into a hole!")
            break
        elif truncated:
            print("⏰ Time limit reached!")
            break
        
        print(describe_state())
    
    env.close()

text_based_navigation()
```

## Advanced Environment Usage Patterns

### Parallel Environment Processing

```python
import concurrent.futures
from rllm.environments import SingleTurnEnvironment
from rllm.environments.env_utils import parallel_task_manager

def batch_question_answering():
    """Example of processing multiple environments in parallel."""
    
    # Create multiple tasks
    tasks = [
        {"question": f"What is {i} * {i}?", "ground_truth": str(i*i)}
        for i in range(1, 11)
    ]
    
    def process_single_task(task_data):
        env = SingleTurnEnvironment(task=task_data)
        obs, info = env.reset()
        
        # Simulate agent response
        question = obs["question"]
        # Extract numbers and compute answer
        import re
        numbers = re.findall(r'\d+', question)
        if len(numbers) >= 2:
            result = int(numbers[0]) * int(numbers[1])
            answer = f"The answer is {result}"
        else:
            answer = "I don't know"
        
        _, reward, _, _, _ = env.step(answer)
        env.close()
        
        return reward
    
    # Process tasks in parallel
    with parallel_task_manager(process_single_task, tasks, max_workers=4) as results:
        total_reward = sum(result for _, result in results)
        print(f"Processed {len(results)} tasks with total reward: {total_reward}")

batch_question_answering()
```

### Environment Configuration Management

```python
import json
from rllm.environments import SingleTurnEnvironment, ToolEnvironment

def environment_configuration_example():
    """Example of using JSON configurations for reproducible experiments."""
    
    # Define configurations
    configs = {
        "math_qa": {
            "task": {"question": "What is the integral of x^2?", "ground_truth": "x^3/3 + C"},
            "max_steps": 1
        },
        "tool_usage": {
            "task": {"question": "Calculate the area of a circle with radius 5"},
            "tools": ["calculator"],
            "max_steps": 5
        }
    }
    
    # Save configurations
    with open("env_configs.json", "w") as f:
        json.dump(configs, f, indent=2)
    
    # Load and use configurations
    with open("env_configs.json", "r") as f:
        loaded_configs = json.load(f)
    
    # Create environments from configs
    math_env = SingleTurnEnvironment.from_json(loaded_configs["math_qa"])
    tool_env = ToolEnvironment.from_json(loaded_configs["tool_usage"])
    
    # Test the environments
    for name, env in [("Math", math_env), ("Tool", tool_env)]:
        obs, info = env.reset()
        print(f"{name} Environment - Question: {obs.get('question', 'N/A')}")
        env.close()

environment_configuration_example()
```

## Best Practices

### Error Handling and Robustness

```python
from rllm.environments import ToolEnvironment
import logging

def robust_environment_usage():
    """Example of robust environment usage with proper error handling."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    env = None
    try:
        env = ToolEnvironment(
            task={"question": "Calculate 10 factorial"},
            tools=["calculator"],
            max_steps=3
        )
        
        obs, info = env.reset()
        logger.info(f"Environment initialized successfully: {obs}")
        
        # Simulate potential error conditions
        tool_calls = [{
            "id": "calc_1",
            "function": {
                "name": "calculator",
                "arguments": '{"expr": "10!"}'  # May not be supported
            }
        }]
        
        try:
            next_obs, reward, done, info = env.step(tool_calls)
            logger.info(f"Tool execution successful: {reward}")
        except Exception as tool_error:
            logger.warning(f"Tool execution failed: {tool_error}")
            # Fallback strategy
            fallback_calls = [{
                "id": "calc_2",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expr": "10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1"}'
                }
            }]
            next_obs, reward, done, info = env.step(fallback_calls)
            logger.info(f"Fallback successful: {reward}")
            
    except Exception as e:
        logger.error(f"Environment error: {e}")
        return False
    finally:
        if env:
            env.close()
            logger.info("Environment closed successfully")
    
    return True

success = robust_environment_usage()
print(f"Robust execution completed: {success}")
```

### Performance Monitoring

```python
import time
from rllm.environments import SingleTurnEnvironment

def performance_monitoring_example():
    """Example of monitoring environment performance."""
    
    start_time = time.time()
    
    # Create environment
    env = SingleTurnEnvironment(
        task={"question": "What is machine learning?"}
    )
    
    init_time = time.time() - start_time
    
    # Measure reset performance
    reset_start = time.time()
    obs, info = env.reset()
    reset_time = time.time() - reset_start
    
    # Measure step performance
    step_start = time.time()
    answer = "Machine learning is a subset of AI that enables computers to learn patterns from data."
    next_obs, reward, terminated, truncated, info = env.step(answer)
    step_time = time.time() - step_start
    
    # Measure cleanup
    close_start = time.time()
    env.close()
    close_time = time.time() - close_start
    
    total_time = time.time() - start_time
    
    print(f"Performance Metrics:")
    print(f"  Initialization: {init_time:.4f}s")
    print(f"  Reset: {reset_time:.4f}s") 
    print(f"  Step: {step_time:.4f}s")
    print(f"  Close: {close_time:.4f}s")
    print(f"  Total: {total_time:.4f}s")
    print(f"  Reward achieved: {reward}")

performance_monitoring_example()
```

These examples demonstrate the full range of rLLM environment capabilities, from simple question-answering to complex software engineering tasks. Each environment is designed to support different aspects of LLM training and evaluation, providing rich interaction patterns and realistic task scenarios. 