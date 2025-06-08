# Core Concepts Overview

rLLM is built around several core components that work together to enable reinforcement learning for language models. This page provides a high-level overview of these components and how they interact.

## Architecture

The rLLM architecture consists of the following main components:

![rLLM Architecture](../assets/rllm_architecture.png)

1. **Agents**: LLM-based agents that generate actions based on observations
2. **Environments**: Task-specific environments that provide observations and rewards
3. **Agent Execution Engine**: Orchestrates interactions between agents and environments
4. **Trainer**: Implements RL algorithms to update agent policies based on rewards

## Key Components

### Agents

Agents are responsible for generating actions based on observations from the environment. In rLLM, agents are typically language models wrapped with task-specific prompts and interfaces. Agents implement the `Agent` interface, which defines methods for observation processing, action generation, and internal state updates.

### Environments

Environments define tasks and provide observations and rewards to agents. They implement the `BaseEnv` interface, which follows a similar structure to OpenAI Gym environments. Environments can be:

- **Single-turn**: For tasks requiring only one interaction (e.g., question answering)
- **Multi-turn**: For tasks requiring multiple interactions (e.g., dialogue, web navigation)

### Agent Execution Engine

The Agent Execution Engine manages the interaction between agents and environments. It handles:

- **Agent-environment communication**: Passing observations and actions
- **Trajectory tracking**: Recording sequences of interactions
- **Parallelization**: Running multiple agent-environment pairs simultaneously

### Trainer

The Trainer implements reinforcement learning algorithms to update agent policies based on collected trajectories and rewards. It supports:

- **PPO (Proximal Policy Optimization)**: The primary RL algorithm used in rLLM
- **Distributed training**: Parallelized training across multiple machines
- **Evaluation**: Tools for assessing agent performance during and after training

## Data Flow

The typical data flow in rLLM follows these steps:

1. The **Environment** generates an initial observation
2. The **Agent** processes the observation and generates an action
3. The **Agent Execution Engine** orchestrate the interaction between agent and environment to collect trajectories.
6. The **Trainer** uses these trajectories to update the agent's policy

This cycle repeats until training is complete or the environment signals completion.

## Next Steps

To learn more about each component, explore the dedicated pages:

- [Agents](agents.md)
- [Environments](environments.md)
- [Agent Execution Engine](execution-engine.md)
- [Training](training.md) 