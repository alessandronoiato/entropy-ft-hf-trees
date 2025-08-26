import argparse
from dataclasses import dataclass
from typing import List

import os
import sys

# Ensure project root is on sys.path when running as a script (python scripts/run_ppo_trl.py)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from transformers import GPT2TokenizerFast, GenerationConfig
from datasets import Dataset
from trl import PPOConfig, PPOTrainer
from trl.trainer import utils as trl_utils

from tokenization.tree_tokenizer import build_tree_tokenizer
from env.tree_env import UpDownTree, TreeConfig
from utils.masking import TreeLogitsProcessor
from models.build import ModelCfg, build_models_for_trl21
import json
import math
import csv


@dataclass
class TrainArgs:
    horizon: int
    steps: int
    batch_size: int
    mini_batch_size: int
    learning_rate: float
    kl_coef: float
    seed: int


def compute_sequence_logprobs(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    tokenizer,
    env,
    id_U: int,
    id_D: int,
    id_EOS: int,
) -> torch.Tensor:
    # returns masked log p(seq) including BOS->...->EOS, summing next-token logprobs with legality renormalization
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]
        next_logits = logits[:, :-1, :]
        next_ids = input_ids[:, 1:]
        B, S, V = next_logits.shape
        device = next_logits.device
        token_logps = torch.zeros((B, S), dtype=next_logits.dtype, device=device)
        for b in range(B):
            seq_tokens = input_ids[b].tolist()
            # iterate positions t (next-token index is t)
            for t in range(S):
                # Build prefix actions up to current position (excluding BOS/EOS/PAD)
                prefix_ids = seq_tokens[: t + 1]
                prefix_actions = [i for i in prefix_ids if i not in {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}]
                allow_U, allow_D, allow_EOS = env.legal_actions(prefix_actions, id_U, id_D, id_EOS)
                allowed_ids = []
                if allow_U:
                    allowed_ids.append(id_U)
                if allow_D:
                    allowed_ids.append(id_D)
                if allow_EOS:
                    allowed_ids.append(id_EOS)
                # If no allowed tokens (shouldn't happen), skip
                if len(allowed_ids) == 0:
                    continue
                logits_bt = next_logits[b, t]
                allowed_logits = logits_bt[allowed_ids]
                logZ = torch.logsumexp(allowed_logits, dim=-1)
                next_id = next_ids[b, t].item()
                logp = logits_bt[next_id] - logZ
                # mask out PAD positions (use next-token mask)
                if attention_mask[b, t + 1].item() == 1:
                    token_logps[b, t] = logp
        seq_logp = token_logps.sum(dim=-1)
        return seq_logp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mini_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_samples", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Tokenizer
    tokenizer = build_tree_tokenizer()
    id_U = tokenizer.convert_tokens_to_ids(["U"])[0]
    id_D = tokenizer.convert_tokens_to_ids(["D"])[0]
    id_EOS = tokenizer.eos_token_id

    # Env and logits processor
    env = UpDownTree(TreeConfig(horizon=args.horizon))
    logits_processor = TreeLogitsProcessor(env, tokenizer, id_U, id_D, id_EOS)

    # Models (policy, value, ref, reward) for TRL 0.21
    model_cfg = ModelCfg(vocab_size=len(tokenizer))
    policy, value_model, ref_model, reward_model = build_models_for_trl21(model_cfg)

    # TRL 0.21 config: per-device batch size and mini-batches are handled via these fields
    ppo_cfg = PPOConfig(
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_mini_batches=max(1, args.batch_size // args.mini_batch_size),
        num_ppo_epochs=4,
        kl_coef=args.kl_coef,
        report_to=[],
        seed=args.seed,
        fp16=False,
        bf16=False,
        use_mps_device=torch.backends.mps.is_available(),
        use_cpu=(not torch.cuda.is_available() and not torch.backends.mps.is_available()),
        push_to_hub=False,
        response_length=args.horizon + 1,
        local_rollout_forward_batch_size=args.batch_size,
        stop_token_id=tokenizer.eos_token_id,
        per_device_eval_batch_size=args.batch_size,
        num_sample_generations=0,
        total_episodes=args.steps * args.batch_size,
    )

    # Build a trivial dataset of BOS prompts of fixed length 1 that TRL will iterate
    prompts = [[tokenizer.bos_token_id] for _ in range(args.batch_size)]
    train_ds = Dataset.from_dict({"input_ids": prompts, "attention_mask": [[1]] * len(prompts)})

    # Construct PPOTrainer per TRL 0.21 signature
    trainer = PPOTrainer(
        args=ppo_cfg,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=train_ds,
        eval_dataset=train_ds,
        value_model=value_model,
    )

    # Training loop
    # Inject tree masking during generation by patching TRL's generate() helper
    def _patched_generate(lm_backbone, queries, pad_token_id, generation_config):
        context_length = queries.shape[1]
        attention_mask = queries != pad_token_id
        input_ids = torch.masked_fill(queries, ~attention_mask, 0)
        output = lm_backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=[logits_processor],
        )
        logits = torch.stack(output.scores, 1)
        return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits

    trl_utils.generate = _patched_generate

    # Run training loop via TRL's internal trainer (handles generation, KL, rewards, updates)
    trainer.train()

    # Evaluation: sample sequences and compute reward, entropy, KL, and Up/Down (p, q) params
    with torch.no_grad():
        num = args.eval_samples
        device = trainer.accelerator.device
        queries = torch.tensor([[tokenizer.bos_token_id]] * num, dtype=torch.long, device=device)
        gen_cfg = GenerationConfig(
            max_new_tokens=args.horizon + 1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_k=0,
        )
        out = policy.generate(
            inputs=queries,
            generation_config=gen_cfg,
        logits_processor=[logits_processor],
            return_dict_in_generate=True,
        )
        responses = out.sequences.to(device)
        attention_mask = (responses != tokenizer.pad_token_id).long().to(device)
        seq_logp_ref = compute_sequence_logprobs(ref_model, responses, attention_mask, tokenizer, env, id_U, id_D, id_EOS)
        seq_logp_pol = compute_sequence_logprobs(policy, responses, attention_mask, tokenizer, env, id_U, id_D, id_EOS)
        mean_reward = (-seq_logp_ref).mean().item()
        approx_kl = (seq_logp_pol - seq_logp_ref).mean().item()
        seq_entropy = (-seq_logp_pol).mean().item()

        # Token-level probabilities for U/D to estimate p and q
        outputs = policy(input_ids=responses, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # next-token logits
        B, S, V = logits.shape
        # Compute p(U | history) renormalized among {U,D} only, respecting legality
        pU_renorm = torch.zeros((B, S), device=device, dtype=logits.dtype)
        both_allowed = torch.zeros((B, S), dtype=torch.bool, device=device)
        first_is_D = torch.zeros((B,), dtype=torch.bool, device=device)
        for b in range(B):
            ids = responses[b].tolist()
            actions = [i for i in ids if i not in {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}]
            if len(actions) > 0 and actions[0] == id_D:
                first_is_D[b] = True
            for s in range(S):
                prefix_ids = responses[b, : s + 1].tolist()
                prefix_actions = [i for i in prefix_ids if i not in {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}]
                allow_U, allow_D, _ = env.legal_actions(prefix_actions, id_U, id_D, id_EOS)
                both_allowed[b, s] = bool(allow_U and allow_D)
                # compute softmax over allowed {U,D}
                logits_bs = logits[b, s]
                terms = []
                if allow_U:
                    terms.append(logits_bs[id_U])
                if allow_D:
                    terms.append(logits_bs[id_D])
                if len(terms) == 0:
                    continue
                # logsumexp for {U,D}
                logZ_ud = torch.logsumexp(torch.stack(terms), dim=-1)
                if allow_U:
                    pU_renorm[b, s] = torch.exp(logits_bs[id_U] - logZ_ud)

        # p: probability of choosing U at first step (renormalized among {U,D})
        p_param = pU_renorm[:, 0].mean().item()
        # q: conditional U probability across steps where first action was D and both actions allowed
        # only positions after the first action (s >= 1)
        pos_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        after_first = pos_idx >= 1
        mask_q = (first_is_D.unsqueeze(1) & both_allowed & after_first)
        q_values = pU_renorm[mask_q]
        q_param = q_values.mean().item() if q_values.numel() > 0 else float("nan")

        # Sample-based estimates
        first_actions = responses[:, 1]
        p_sample = (first_actions == id_U).float().mean().item()
        # among trajectories with first D, fraction of U in subsequent actions
        q_counts = []
        for b in range(B):
            ids = responses[b].tolist()
            acts = [i for i in ids if i not in {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}]
            if len(acts) == 0 or acts[0] != id_D:
                continue
            if len(acts) > 1:
                subsequent = acts[1:]
                if len(subsequent) > 0:
                    u_frac = sum(1 for a in subsequent if a == id_U) / float(len(subsequent))
                    q_counts.append(u_frac)
        q_sample = float(sum(q_counts) / len(q_counts)) if q_counts else float("nan")

        # Analytic proxy for sequence entropy using p and q (ignores EOS details)
        def H_binary(x: float) -> float:
            eps = 1e-8
            x = min(max(x, eps), 1.0 - eps)
            return float(-(x * torch.log(torch.tensor(x)) + (1 - x) * torch.log(torch.tensor(1 - x))))

        analytic_seq_entropy = H_binary(p_param) + (1.0 - p_param) * max(args.horizon - 1, 0) * H_binary(q_param) if q_param == q_param else float("nan")
        # first action stats
        first_actions = responses[:, 1]
        frac_U = (first_actions == id_U).float().mean().item()
        frac_D = (first_actions == id_D).float().mean().item()
        # eos coverage
        has_eos = (responses == tokenizer.eos_token_id).any(dim=1).float().mean().item()
        print({
            "eval/mean_reward": mean_reward,
            "eval/approx_kl": approx_kl,
            "eval/seq_entropy": seq_entropy,
            "eval/frac_U": frac_U,
            "eval/frac_D": frac_D,
            "eval/has_eos_frac": has_eos,
            "eval/p_param": p_param,
            "eval/q_param": q_param,
            "eval/p_sample": p_sample,
            "eval/q_sample": q_sample,
            "eval/analytic_seq_entropy_proxy": analytic_seq_entropy,
        })

    # Exact entropy over all legal sequences (enumeration for small H)
    def compute_exact_sequence_probs(model, tokenizer, env, id_U, id_D, id_EOS):
        device = next(model.parameters()).device
        bos = tokenizer.bos_token_id
        prefixes = [([bos], [], 0.0)]  # (token_ids, actions_wo_specials, logp)
        finals = []  # list of (action_str, prob)
        model.eval()
        with torch.no_grad():
            while prefixes:
                next_prefixes = []
                for token_ids, actions, logp_sum in prefixes:
                    allow_U, allow_D, allow_EOS = env.legal_actions(actions, id_U, id_D, id_EOS)
                    allowed_ids = []
                    if allow_U:
                        allowed_ids.append(id_U)
                    if allow_D:
                        allowed_ids.append(id_D)
                    if allow_EOS:
                        allowed_ids.append(id_EOS)
                    if not allowed_ids:
                        continue
                    inp = torch.tensor([token_ids], dtype=torch.long, device=device)
                    attn = torch.ones_like(inp)
                    logits = model(input_ids=inp, attention_mask=attn).logits[0, -1]
                    logZ = torch.logsumexp(logits[allowed_ids], dim=-1).item()
                    for tok in allowed_ids:
                        logp_tok = (logits[tok].item() - logZ)
                        new_logp = logp_sum + logp_tok
                        if tok == id_EOS:
                            actions_str = "".join([tokenizer.convert_ids_to_tokens([a])[0] for a in actions])
                            finals.append((actions_str, math.exp(new_logp)))
                        else:
                            new_actions = actions + [tok]
                            new_tokens = token_ids + [tok]
                            next_prefixes.append((new_tokens, new_actions, new_logp))
                prefixes = next_prefixes
        total_p = sum(p for _, p in finals)
        if total_p > 0:
            finals = [(a, p / total_p) for a, p in finals]
        H = -sum(p * math.log(p + 1e-40) for _, p in finals)
        return H, finals

    exact_H, seqs = compute_exact_sequence_probs(policy, tokenizer, env, id_U, id_D, id_EOS)
    report = {
        "horizon": args.horizon,
        "num_sequences": len(seqs),
        "entropy_nats": exact_H,
        "theoretical_max_nats": math.log(2 + 3 * (2 ** (args.horizon - 2))),
        "sum_probs": sum(p for _, p in seqs),
    }
    os.makedirs(os.path.join(_PROJECT_ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(_PROJECT_ROOT, "outputs", "ppo_exact_entropy.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(_PROJECT_ROOT, "outputs", "ppo_sequence_probs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "probability"])
        for a, p in sorted(seqs, key=lambda x: -x[1]):
            w.writerow([a, p])


if __name__ == "__main__":
    main()

