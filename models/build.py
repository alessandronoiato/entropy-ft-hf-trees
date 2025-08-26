from dataclasses import dataclass
from typing import Tuple

import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel


@dataclass
class ModelCfg:
    vocab_size: int
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    max_position_embeddings: int = 64


def build_policy_and_ref(cfg: ModelCfg) -> Tuple[GPT2LMHeadModel, GPT2LMHeadModel]:
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
    policy = GPT2LMHeadModel(gpt_cfg)
    # Optionally load a local state dict (models/pretrained.pt or ENTROPY_PRETRAINED_PATH)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_pt = os.path.join(project_root, "models", "pretrained.pt")
    pt_path = os.environ.get("ENTROPY_PRETRAINED_PATH", default_pt)
    if os.path.isfile(pt_path):
        state = torch.load(pt_path, map_location="cpu")
        policy.load_state_dict(state, strict=False)
    ref = GPT2LMHeadModel(gpt_cfg)
    ref.load_state_dict(policy.state_dict(), strict=True)
    for p in ref.parameters():
        p.requires_grad_(False)
    return policy, ref


