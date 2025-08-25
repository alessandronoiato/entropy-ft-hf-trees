from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel


@dataclass
class ModelCfg:
    vocab_size: int
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    max_position_embeddings: int = 64


class CachingGPT2LMHeadModel(GPT2LMHeadModel):
    """GPT-2 LM that caches last input_ids for reward/value computation."""

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self._last_input_ids: torch.Tensor | None = None

    def forward(self, *args, **kwargs):  # type: ignore[override]
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        if input_ids is not None:
            self._last_input_ids = input_ids
        return super().forward(*args, **kwargs)


class ValueHeadModel(nn.Module):
    """Minimal value head model compatible with TRL 0.21 expectations.

    Exposes a `base_model_prefix` attribute pointing to the LM backbone and a
    `.score(hidden_states)` method returning per-token scalar values [B, T, 1].
    """

    base_model_prefix: str = "transformer"

    def __init__(self, backbone: GPT2LMHeadModel):
        super().__init__()
        # Clone the backbone so value and policy can diverge
        self.transformer = CachingGPT2LMHeadModel(backbone.config)
        self.transformer.load_state_dict(backbone.state_dict(), strict=True)
        self.value_head = nn.Linear(backbone.config.n_embd, 1, bias=False)

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, T, H] -> [B, T, 1]
        return self.value_head(hidden_states)


class SequenceNLLRewardModel(nn.Module):
    """Reward model that outputs cumulative negative log-likelihood under a frozen reference LM.

    The `.score(hidden_states)` returns a tensor of shape [B, T, 1] where the
    last valid timestep equals the sequence-level self-surprise -log p_ref(seq).
    """

    base_model_prefix: str = "transformer"

    def __init__(self, ref_backbone: GPT2LMHeadModel):
        super().__init__()
        # Use a caching backbone initialized from the provided reference model
        self.transformer = CachingGPT2LMHeadModel(ref_backbone.config)
        self.transformer.load_state_dict(ref_backbone.state_dict(), strict=True)
        # Freeze reward model
        for p in self.parameters():
            p.requires_grad_(False)

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, T, H]
        device = hidden_states.device
        lm_head = self.transformer.lm_head  # tied to embeddings
        logits = F.linear(hidden_states, lm_head.weight, lm_head.bias)  # [B, T, V]
        # next-token ids from cached inputs
        input_ids = self.transformer._last_input_ids  # [B, T]
        if input_ids is None:
            raise RuntimeError("RewardModel backbone did not cache input_ids. Ensure forward() was called.")
        # compute NLL for next tokens at each position (except final position)
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
        next_ids = input_ids[:, 1:].unsqueeze(-1)  # [B, T-1, 1]
        token_logp = torch.gather(logprobs, dim=-1, index=next_ids).squeeze(-1)  # [B, T-1]
        token_nll = -token_logp  # [B, T-1]
        # cumulative NLL aligned to positions (prefix sum; index t holds sum of first t next-tokens)
        prefix = torch.zeros(token_nll.size(0), 1, device=device, dtype=token_nll.dtype)
        cumsum = torch.cumsum(token_nll, dim=-1)
        reward_logits = torch.cat([prefix, cumsum], dim=-1)  # [B, T]
        return reward_logits.unsqueeze(-1)  # [B, T, 1]


def build_models_for_trl21(cfg: ModelCfg) -> Tuple[GPT2LMHeadModel, ValueHeadModel, GPT2LMHeadModel, SequenceNLLRewardModel]:
    gpt_cfg = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        n_positions=cfg.max_position_embeddings,
        n_ctx=cfg.max_position_embeddings,
        bos_token_id=1,
        eos_token_id=2,
    )
    # Policy LM (trainable)
    policy = GPT2LMHeadModel(gpt_cfg)
    # Reference LM for KL (frozen)
    ref = GPT2LMHeadModel(gpt_cfg)
    ref.load_state_dict(policy.state_dict(), strict=True)
    for p in ref.parameters():
        p.requires_grad_(False)
    # Value model (trainable head)
    value_model = ValueHeadModel(policy)
    # Reward model using the frozen ref backbone (frozen)
    reward_model = SequenceNLLRewardModel(ref)
    return policy, value_model, ref, reward_model

