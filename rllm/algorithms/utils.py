
import torch
from typing import Optional


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()
    

def masked_sum(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute sum of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis)
    else:
        return (values * mask).sum()
    

def get_action_logprobs(model, obs_tokens, action_tokens):
    input_ids = torch.cat([obs_tokens.input_ids, action_tokens.input_ids], dim=1)
    attention_mask = torch.cat([obs_tokens.attention_mask, action_tokens.attention_mask], dim=1)

    obs_length = obs_tokens.input_ids.size(1)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, obs_length-1:-1, :]
    targets = action_tokens.input_ids

    logprobs_all = torch.log_softmax(logits, dim=-1)
    logprobs = torch.gather(logprobs_all, -1, targets.unsqueeze(-1)).squeeze(-1)

    return logprobs * action_tokens.attention_mask