# Core Concepts Overview

rLLM is built around several core components that work together to enable reinforcement learning for language models. This page provides a high-level overview of these components and how they interact.

The rLLM architecture consists of the following main components:

1. **Agents**: LLM-based agents that generate actions based on environment observations
2. **Environments**: Task-specific environments that provide observations and rewards
3. **Agent Execution Engine**: Orchestrates interactions between agents and environments
4. **Trainer**: RL algorithms to update agent policies based on rewards

## Key Components

### Agents

Agents are responsible for generating actions based on observations from the environment. In rLLM, agents are typically language models wrapped with task-specific prompts and interfaces. Agents implement the `Agent` interface, which defines methods for observation processing, action generation, and internal state updates.

### Environments

Environments define tasks and provide observations and rewards to agents. They implement the `BaseEnv` interface, which follows a similar structure to OpenAI Gym environments.

### AgentExecutionEngine

The AgentExecutionEngine manages the interaction between agents and environments. It handles:

- **Agent-environment interaction**: Passing observations and actions
- **Async Parallel Rollout**: Running multiple agent-environment pairs simultaneously and asynchronously
- **Integration with training backend**: The agent execution engine handles trajectory rollout for RL integration

### AgentTrainer

The AgentTrainer implements reinforcement learning algorithms to update agent policies based on collected trajectories and rewards. We use `verl` as our training backend, and integrates our `AgentExecutionEngine` for trajectory rollout.  

## Next Steps

To learn more about each component, explore the dedicated pages:

- [Agents](agents.md)
- [Environments](environments.md)
- [Agent Execution Engine](execution-engine.md)
- [Training](training.md) 