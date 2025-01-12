from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
import wandb
from accelerate import Accelerator

from .core import get_action_logprobs

@dataclass
class ReinforceConfig:
    lr: float = 1e-6
    kl_coef: float = 0.01
    kl_penalty: str = "kl"
    gradient_accumulation_steps: int = 1
    use_wandb: bool = False


class ReinforceTrainer:
    def __init__(self, model, tokenizer, config: ReinforceConfig, ref_model: Optional[torch.nn.Module] = None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.accelerator = Accelerator(gradient_accumulation_steps=self.config.gradient_accumulation_steps)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare(self.ref_model)
        self.device = self.accelerator.device

        self.step_counter = 0

        if self.ref_model is not None:
            self.ref_model.eval()

        if self.config.use_wandb:
            if self.accelerator.is_main_process:
                wandb.init()

        self.step_counter = 0
    
    def step(self, sample_batch):
        obs = sample_batch["queries"]
        actions = sample_batch["responses"]
        scores = sample_batch["rewards"]
        
        obs_tokens = self.tokenizer(obs, return_tensors="pt", padding=True, padding_side="left").to(self.model.device)
        action_tokens = self.tokenizer(actions, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False).to(self.model.device)

        logprobs = get_action_logprobs(self.model, obs_tokens, action_tokens)
        masks = action_tokens.attention_mask

        kl_losses = None
        if self.ref_model is not None:
            with torch.no_grad():
                logprobs_ref = get_action_logprobs(self.ref_model, obs_tokens, action_tokens)
                kl_losses = logprobs - logprobs_ref
                kl_losses = (kl_losses * masks).sum(dim=-1)
        
        scores = torch.tensor(scores).to(self.model.device)
        rewards = scores
        if kl_losses is not None:
            rewards -= self.config.kl_coef * kl_losses

        logprobs = (logprobs * masks).sum(dim=-1)
        policy_loss = torch.mean(-(logprobs * rewards))

        policy_loss = policy_loss / self.config.gradient_accumulation_steps
        self.accelerator.backward(policy_loss)
    
        if (self.step_counter + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.step_counter += 1

        if kl_losses is not None:
            kl_mean = self.accelerator.gather(kl_losses).mean().item()
        else:
            kl_mean = None

        stats = {
            "loss": self.accelerator.gather(policy_loss).mean().item(),
            "reward": self.accelerator.gather(rewards).mean().item(),
            "scores": self.accelerator.gather(scores).mean().item(),
            "kl": kl_mean
        }

        if wandb.run is not None and self.config.use_wandb:
            df = pd.DataFrame({
                "obs": obs,
                "action": actions, 
                "score": scores.cpu()
            })
            table = wandb.Table(dataframe=df)
            wandb.log({"samples": table}, commit=False)
            wandb.log(stats)

        return stats

    def save(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)