import argparse
import os
import sys
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tokenization.tree_tokenizer import build_tree_tokenizer
from env.tree_env import UpDownTree, TreeConfig
from utils.masking import TreeLogitsProcessor


class TreeSequences(Dataset):
    def __init__(self, tokenizer, env: UpDownTree, num_samples: int, horizon: int):
        self.tokenizer = tokenizer
        self.env = env
        self.num_samples = num_samples
        self.horizon = horizon
        self.id_U = tokenizer.convert_tokens_to_ids(["U"])[0]
        self.id_D = tokenizer.convert_tokens_to_ids(["D"])[0]
        self.id_EOS = tokenizer.eos_token_id

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # Sample a legal sequence by uniformly sampling allowed actions
        ids: List[int] = [self.tokenizer.bos_token_id]
        actions: List[int] = []
        for t in range(self.horizon + 1):
            allow_U, allow_D, allow_EOS = self.env.legal_actions(actions, self.id_U, self.id_D, self.id_EOS)
            choices = []
            if allow_U:
                choices.append(self.id_U)
            if allow_D:
                choices.append(self.id_D)
            if allow_EOS:
                choices.append(self.id_EOS)
            next_id = choices[torch.randint(len(choices), (1,)).item()]
            ids.append(next_id)
            if next_id == self.id_EOS:
                break
            actions.append(next_id)
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def collate_pad(batch, pad_id: int):
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        L = ex["input_ids"].size(0)
        input_ids[i, :L] = ex["input_ids"]
        attention_mask[i, :L] = 1
    labels = input_ids.clone()
    labels[labels == pad_id] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--out", default=os.path.join(_PROJECT_ROOT, "models", "pretrained.pt"))
    args = parser.parse_args()

    tokenizer = build_tree_tokenizer()
    env = UpDownTree(TreeConfig(horizon=args.horizon))

    cfg = GPT2Config(
        vocab_size=len(tokenizer),
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_positions=64,
        n_ctx=64,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(cfg)

    ds = TreeSequences(tokenizer, env, num_samples=args.samples, horizon=args.horizon)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_pad(b, tokenizer.pad_token_id))

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    step = 0
    for epoch in range(max(1, args.steps // max(1, len(dl)))):
        for batch in dl:
            step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            if step >= args.steps:
                break
        if step >= args.steps:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved pretrained state_dict to {args.out}")


if __name__ == "__main__":
    main()


