# rLLM: Reinforcement Learning for Language Agents

rLLM is a comprehensive framework for training language agents using reinforcement learning. It enables you to easily define custom language agents and environments, collect agent trajectories, and perform RL training to continuously improve your agents' performance.

## Key Features

rLLM provides:
- **Simple abstractions for building custom agents**: rLLM decouples agent and environment abstractions from the underlying training infrastructure. Users can easily define custom agents and train them with RL without getting entangled in the complexities of the underlying training engine.
- **Unified interface for agent inference & training**: Training and deploying LLM agents traditionally requires two separate sets of tooling and infrastructure. rLLM provides a unified interface for both training and deploying language agents, enabling continuous evolution and training of agents to "learn from experience."
- **Efficient trajectory generation & scalable RL training**: rLLM's execution engine supports asynchronous and parallelized generation of agent trajectories. For RL training, rLLM integrates `verl` as its training backend, which supports scalable RL training for language models. Together, rLLM delivers efficient and scalable training for language agents.


## Getting Started

To get started with rLLM, check out the [Installation Guide](getting-started/installation.md) and [Quick Start Tutorial](getting-started/quick-start.md).

## Built-in Agents

rLLM currently supports a variety of built-in agents:
- **Math/Coding Agents**: Train single-turn reasoning models for competition math and coding (like DeepScaleR and DeepCoder), or multi-turn math/coding agents that can iteratively refine their previous answers.
- **SWEAgent and SWEEnv**: Train SWEAgents that can write software patches and resolve real-world GitHub issues.
- **Web Agents**: Train LLMs to navigate websites and perform complex web tasks.
- **General Tool-Using Agents & MCP Environment**: Connect to any MCP servers and train language agents to effectively use tools from the Model Context Protocol (MCP).

## Build Your Own Agents & Environments

Walk through our documentation and examples to understand the fundamentals of rLLM and build your own custom agents and environments tailored to your specific use cases.

## Community & Support

rLLM is an open-source project under active development. We welcome contributions, bug reports, and feature requests from the community.