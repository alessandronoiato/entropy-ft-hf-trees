import argparse
from typing import List

import os
import sys

# Ensure project root is on sys.path when running as a script (python scripts/run_grpo_trl.py)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from datasets import Dataset
from transformers import GenerationConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from transformers.trainer_callback import TrainerCallback

from tokenization.tree_tokenizer import build_tree_tokenizer
from env.tree_env import UpDownTree, TreeConfig
from utils.masking import TreeLogitsProcessor
from models.build import ModelCfg, build_policy_and_ref
import json
import math
import csv


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
            for t in range(S):
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
                if len(allowed_ids) == 0:
                    continue
                logits_bt = next_logits[b, t]
                allowed_logits = logits_bt[allowed_ids]
                logZ = torch.logsumexp(allowed_logits, dim=-1)
                next_id = next_ids[b, t].item()
                logp = logits_bt[next_id] - logZ
                if attention_mask[b, t + 1].item() == 1:
                    token_logps[b, t] = logp
        seq_logp = token_logps.sum(dim=-1)
        return seq_logp


def compute_exact_sequence_probs(model, tokenizer, env, id_U, id_D, id_EOS):
    device = next(model.parameters()).device
    bos = tokenizer.bos_token_id
    prefixes = [([bos], [], 0.0)]
    finals = []
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


def make_sequence_self_surprise_reward(ref_model, tokenizer, env, id_U: int, id_D: int, id_EOS: int):
    ref_device = next(ref_model.parameters()).device

    def reward_func(prompts: List[str], completions: List[str], completion_ids: List[List[int]], **kwargs) -> List[float]:
        # Build input_ids by concatenating tokenized prompt and provided completion ids (already EOS-truncated)
        prompt_ids_list: List[List[int]] = []
        for p in prompts:
            ids = tokenizer(text=p, return_tensors=None, add_special_tokens=False).get("input_ids", None)
            if ids is None:
                # PreTrainedTokenizerFast returns a list when return_tensors=None
                ids = tokenizer(p, add_special_tokens=False)["input_ids"]
            prompt_ids_list.append(ids)

        batch_ids: List[torch.Tensor] = []
        for pid, cid in zip(prompt_ids_list, completion_ids):
            seq = torch.tensor(pid + cid, dtype=torch.long)
            batch_ids.append(seq)

        # Pad and move to device
        max_len = max(seq.size(0) for seq in batch_ids)
        input_ids = torch.full((len(batch_ids), max_len), fill_value=tokenizer.pad_token_id, dtype=torch.long)
        for i, seq in enumerate(batch_ids):
            input_ids[i, : seq.size(0)] = seq
        input_ids = input_ids.to(ref_device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(ref_device)

        # Compute sequence log-prob under reference model
        seq_logp_ref = compute_sequence_logprobs(
            ref_model, input_ids, attention_mask, tokenizer, env, id_U, id_D, id_EOS
        )
        rewards = (-seq_logp_ref).float().cpu().tolist()
        return rewards

    return reward_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=64, help="Number of optimizer steps (max_steps)")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.05, help="KL coefficient for GRPO")
    parser.add_argument("--num_generations", type=int, default=4, help="Completions per prompt (must divide batch)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_samples", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Tokenizer and environment
    tokenizer = build_tree_tokenizer()
    id_U = tokenizer.convert_tokens_to_ids(["U"])[0]
    id_D = tokenizer.convert_tokens_to_ids(["D"])[0]
    id_EOS = tokenizer.eos_token_id
    env = UpDownTree(TreeConfig(horizon=args.horizon))
    logits_processor = TreeLogitsProcessor(env, tokenizer, id_U, id_D, id_EOS)

    # Models: policy and frozen ref
    model_cfg = ModelCfg(vocab_size=len(tokenizer))
    policy, ref_model = build_policy_and_ref(model_cfg)

    # Save policy to a local directory so GRPO can reload both policy and ref from the same model_id path
    artifacts_dir = os.path.join(_PROJECT_ROOT, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "grpo_policy_init")
    policy.save_pretrained(model_path)

    # Build GRPO config
    if args.batch_size % args.num_generations != 0:
        raise ValueError("batch_size must be divisible by num_generations for GRPO.")

    grpo_cfg = GRPOConfig(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.steps,
        seed=args.seed,
        fp16=False,
        bf16=False,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        repetition_penalty=1.0,
        max_completion_length=args.horizon + 1,
        num_generations=args.num_generations,
        steps_per_generation=None,
        generation_batch_size=None,
        beta=args.beta,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=1.0,
        remove_unused_columns=False,
        report_to=[],
        use_transformers_paged=False,
        use_vllm=False,
    )

    # Dataset of prompts: use explicit BOS token to start
    prompts = [tokenizer.bos_token] * args.batch_size
    train_ds = Dataset.from_dict({"prompt": prompts})

    # Reward function: sequence-level self-surprise; initial placeholder uses the initial ref
    reward_fn = make_sequence_self_surprise_reward(ref_model, tokenizer, env, id_U, id_D, id_EOS)

    # Trainer
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=train_ds,
        eval_dataset=train_ds,
        processing_class=tokenizer,
    )

    # Inject masking by patching the underlying model's generate method used in GRPO
    import types
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    _orig_generate = unwrapped.generate

    def _generate_with_mask(self, *g_args, **g_kwargs):
        lps = g_kwargs.get("logits_processor", None)
        if lps is None:
            g_kwargs["logits_processor"] = [logits_processor]
        else:
            l = list(lps)
            if logits_processor not in l:
                l = [logits_processor] + l
            g_kwargs["logits_processor"] = l
        return _orig_generate(*g_args, **g_kwargs)

    unwrapped.generate = types.MethodType(_generate_with_mask, unwrapped)

    # Replace reward function to dynamically use the current ref_model (π_k) after trainer init
    def dynamic_reward(prompts, completions, completion_ids, **kwargs):
        ref = trainer.ref_model if trainer.ref_model is not None else trainer.model
        ref_device = next(ref.parameters()).device
        # Build input_ids by concatenating tokenized prompt and provided completion ids
        prompt_ids_list: List[List[int]] = []
        for p in prompts:
            ids = tokenizer(text=p, return_tensors=None, add_special_tokens=False).get("input_ids", None)
            if ids is None:
                ids = tokenizer(p, add_special_tokens=False)["input_ids"]
            prompt_ids_list.append(ids)
        batch_ids: List[torch.Tensor] = []
        for pid, cid in zip(prompt_ids_list, completion_ids):
            seq = torch.tensor(pid + cid, dtype=torch.long)
            batch_ids.append(seq)
        max_len = max(seq.size(0) for seq in batch_ids)
        input_ids = torch.full((len(batch_ids), max_len), fill_value=tokenizer.pad_token_id, dtype=torch.long)
        for i, seq in enumerate(batch_ids):
            input_ids[i, : seq.size(0)] = seq
        input_ids = input_ids.to(ref_device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(ref_device)
        seq_logp_ref = compute_sequence_logprobs(ref, input_ids, attention_mask, tokenizer, env, id_U, id_D, id_EOS)
        rewards = (-seq_logp_ref).float().cpu().tolist()
        return rewards

    trainer.reward_funcs = [dynamic_reward]
    trainer.reward_func_names = ["self_surprise_ref_k"]

    # Callback: log exact entropy over legal sequences at every step
    class ExactEntropyLogger(TrainerCallback):
        def __init__(self, trainer, tokenizer, env, id_U, id_D, id_EOS, out_dir: str = "outputs"):
            self.trainer = trainer
            self.tokenizer = tokenizer
            self.env = env
            self.id_U = id_U
            self.id_D = id_D
            self.id_EOS = id_EOS
            self.out_dir = out_dir
            os.makedirs(self.out_dir, exist_ok=True)
            self.csv_path = os.path.join(self.out_dir, "grpo_entropy_timeseries.csv")
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step", "entropy_nats"])  # header

        def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
            model_eval = self.trainer.accelerator.unwrap_model(self.trainer.model)
            device = next(model_eval.parameters()).device
            bos = self.tokenizer.bos_token_id
            prefixes = [([bos], [], 0.0)]
            finals = []
            model_eval.eval()
            with torch.no_grad():
                while prefixes:
                    next_prefixes = []
                    for token_ids, actions, logp_sum in prefixes:
                        allow_U, allow_D, allow_EOS = self.env.legal_actions(actions, self.id_U, self.id_D, self.id_EOS)
                        allowed_ids = []
                        if allow_U:
                            allowed_ids.append(self.id_U)
                        if allow_D:
                            allowed_ids.append(self.id_D)
                        if allow_EOS:
                            allowed_ids.append(self.id_EOS)
                        if not allowed_ids:
                            continue
                        inp = torch.tensor([token_ids], dtype=torch.long, device=device)
                        attn = torch.ones_like(inp)
                        logits = model_eval(input_ids=inp, attention_mask=attn).logits[0, -1]
                        logZ = torch.logsumexp(logits[allowed_ids], dim=-1).item()
                        for tok in allowed_ids:
                            logp_tok = (logits[tok].item() - logZ)
                            new_logp = logp_sum + logp_tok
                            if tok == self.id_EOS:
                                finals.append(("", math.exp(new_logp)))
                            else:
                                new_actions = actions + [tok]
                                new_tokens = token_ids + [tok]
                                next_prefixes.append((new_tokens, new_actions, new_logp))
                    prefixes = next_prefixes
            total_p = sum(p for _, p in finals)
            if total_p > 0:
                finals = [(a, p / total_p) for a, p in finals]
            H = -sum(p * math.log(p + 1e-40) for _, p in finals)
            # append CSV row only (single time-series file)
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([int(state.global_step), float(H)])

    trainer.add_callback(ExactEntropyLogger(trainer, tokenizer, env, id_U, id_D, id_EOS))

    # Emit BEFORE distribution (initial policy) before training
    def dump_sequence_probs(model, csv_path):
        H, seqs = compute_exact_sequence_probs(model, tokenizer, env, id_U, id_D, id_EOS)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "probability"])
            for a, p in sorted(seqs, key=lambda x: -x[1]):
                w.writerow([a, p])

    dump_sequence_probs(trainer.accelerator.unwrap_model(trainer.model), os.path.join(_PROJECT_ROOT, "outputs", "before_sequence_probs.csv"))

    trainer.train()

    # Simple evaluation: sample sequences and report p, q, entropy like PPO script
    with torch.no_grad():
        device = trainer.accelerator.device
        model_eval = trainer.accelerator.unwrap_model(trainer.model)
        model_eval_ref = (
            trainer.accelerator.unwrap_model(trainer.ref_model)
            if getattr(trainer, "ref_model", None) is not None
            else model_eval
        )
        num = args.eval_samples
        queries = torch.tensor([[tokenizer.bos_token_id]] * num, dtype=torch.long, device=device)
        gen_cfg = GenerationConfig(
            max_new_tokens=args.horizon + 1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_k=0,
        )
        out = model_eval.generate(
            inputs=queries,
            generation_config=gen_cfg,
            logits_processor=[logits_processor],
            return_dict_in_generate=True,
        )
        responses = out.sequences.to(device)
        attention_mask = (responses != tokenizer.pad_token_id).long().to(device)
        seq_logp_ref = compute_sequence_logprobs(model_eval_ref, responses, attention_mask, tokenizer, env, id_U, id_D, id_EOS)
        seq_logp_pol = compute_sequence_logprobs(model_eval, responses, attention_mask, tokenizer, env, id_U, id_D, id_EOS)
        mean_reward = (-seq_logp_ref).mean().item()
        approx_kl = (seq_logp_pol - seq_logp_ref).mean().item()
        seq_entropy = (-seq_logp_pol).mean().item()

        # p, q estimates
        outputs = model_eval(input_ids=responses, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        B, S, V = logits.shape
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
                logits_bs = logits[b, s]
                terms = []
                if allow_U:
                    terms.append(logits_bs[id_U])
                if allow_D:
                    terms.append(logits_bs[id_D])
                if len(terms) == 0:
                    continue
                logZ_ud = torch.logsumexp(torch.stack(terms), dim=-1)
                if allow_U:
                    pU_renorm[b, s] = torch.exp(logits_bs[id_U] - logZ_ud)

        p_param = pU_renorm[:, 0].mean().item()
        pos_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        after_first = pos_idx >= 1
        mask_q = (first_is_D.unsqueeze(1) & both_allowed & after_first)
        q_values = pU_renorm[mask_q]
        q_param = q_values.mean().item() if q_values.numel() > 0 else float("nan")

        first_actions = responses[:, 1]
        frac_U = (first_actions == id_U).float().mean().item()
        frac_D = (first_actions == id_D).float().mean().item()
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
        })

    exact_H, seqs = compute_exact_sequence_probs(model_eval, tokenizer, env, id_U, id_D, id_EOS)
    report = {
        "horizon": args.horizon,
        "num_sequences": len(seqs),
        "entropy_nats": exact_H,
        "theoretical_max_nats": math.log(2 + 3 * (2 ** (args.horizon - 2))),
        "sum_probs": sum(p for _, p in seqs),
    }
    out_dir = os.path.join(_PROJECT_ROOT, "outputs") if "_PROJECT_ROOT" in globals() else "outputs"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "grpo_exact_entropy.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "after_sequence_probs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "probability"])
        for a, p in sorted(seqs, key=lambda x: -x[1]):
            w.writerow([a, p])


if __name__ == "__main__":
    main()


