"""Train the concierge agent with RL using the rLLM Python API.

Assumes register.py has been run first.

Usage:
    python train.py --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse

from rllm.data import DatasetRegistry
from rllm.experimental.cli.train import build_train_config, make_agent_run_func
from rllm.experimental.eval.agent_loader import load_agent
from rllm.experimental.eval.evaluator_loader import load_evaluator
from rllm.experimental.unified_trainer import AgentTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the concierge agent with RL")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name/path")
    parser.add_argument("--group-size", type=int, default=8, help="Rollouts per prompt")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    args = parser.parse_args()

    # Load by name — works because register.py persisted them to ~/.rllm/
    train_dataset = DatasetRegistry.load_dataset("concierge", "train")
    val_dataset = DatasetRegistry.load_dataset("concierge", "test")
    agent = load_agent("concierge")
    evaluator = load_evaluator("relevance")

    agent_run_func = make_agent_run_func(agent, evaluator, args.model)
    config = build_train_config(
        model_name=args.model,
        group_size=args.group_size,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.lora_rank,
        total_epochs=args.epochs,
        total_steps=args.max_steps,
        val_freq=5,
        save_freq=20,
        project="concierge-train",
        experiment="concierge-rl",
        output_dir=None,
        config_file=None,
    )

    trainer = AgentTrainer(
        backend="tinker",
        agent_run_func=agent_run_func,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
