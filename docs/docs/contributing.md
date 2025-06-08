# Contributing to rLLM

We welcome contributions to rLLM! This guide will help you get started with the development process.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package installer)
- CUDA-compatible GPU (recommended for development)

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone --recurse-submodules https://github.com/YOUR-USERNAME/rllm-internal.git
   cd rllm
   ```

3. Install development dependencies:
   ```bash
   pip install -e ./verl[vllm,gpu,sglang]
   pip install -e .
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Additional development dependencies
   ```

## Development Workflow

### Creating a New Feature

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, ensuring you follow the code style guidelines
3. Add tests for your feature
4. Run the tests locally to ensure they pass
5. Commit your changes with a descriptive message
6. Push your branch to GitHub
7. Create a pull request from your branch to the main repository

### Running Tests

To run the tests:

```bash
pytest tests/
```

To run a specific test file:

```bash
pytest tests/test_specific_file.py
```

### Code Style

We use ruff for code style checking and formatting. To check your code:

```bash
ruff check .
```

To automatically format your code:

```bash
ruff format .
```

## Project Structure

Understanding the project structure will help you contribute effectively:

```
rllm/
├── agents/              # Agent implementations
├── data/                # Data handling utilities
├── engine/              # Execution engines
├── environments/        # Environment implementations
├── models/              # Model definitions and utilities
├── parser/              # Parsers for processing text
├── rewards/             # Reward functions
├── router/              # Routing utilities for distributed training
├── tools/               # Tool implementations
├── train/               # Training utilities
└── utils.py             # General utilities
```

## Adding New Components

### Adding a New Agent

1. Create a new file in `rllm/agents/` (e.g., `my_agent.py`)
2. Implement your agent by inheriting from the base `Agent` class
3. Add any necessary system prompts to `rllm/agents/system_prompts.py`
4. Register your agent in `rllm/agents/__init__.py`
5. Add tests in `tests/agents/`

### Adding a New Environment

1. Create a new file in `rllm/environments/` (e.g., `my_env.py`)
2. Implement your environment by inheriting from the base `BaseEnv` class
3. Register your environment in `rllm/environments/__init__.py`
4. Add tests in `tests/environments/`

### Adding a New Reward Function

1. Create a new file in `rllm/rewards/` (e.g., `my_reward.py`)
2. Implement your reward function
3. Register your reward function in `rllm/rewards/__init__.py`
4. Add tests in `tests/rewards/`

## Documentation

When adding new features, please also update the documentation:

1. Add docstrings to your code following the Google Python Style Guide
2. Update or create relevant documentation files in the `docs/` directory
3. If adding a new feature, add an example to the examples section

## Submitting a Pull Request

When you're ready to submit your changes:

1. Make sure your branch is up-to-date with the main branch
2. Ensure all tests pass
3. Check that your code follows the style guidelines
4. Create a pull request with a descriptive title and detailed description
5. Link any relevant issues in your pull request description

## Code Review

After submitting a pull request:

1. Core maintainers will review your code
2. Address any feedback or comments
3. Once approved, your changes will be merged into the main branch

## Communication

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for general questions and ideas
- **Pull Requests**: For submitting code changes

## License

By contributing to rLLM, you agree that your contributions will be licensed under the project's MIT license. 