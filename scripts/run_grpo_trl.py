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
from utils.env_factory import make_env, make_actions, action_ids as make_action_ids
from utils.masking import TreeLogitsProcessor
from utils.sequence_eval import compute_sequence_logprobs, enumerate_sequence_probs, compute_legal_mass_raw
from utils.rollout_mask import apply_masked_generate
from utils.reward import make_self_surprise_reward
from models.build import ModelCfg, build_policy_and_ref
import json
import math
import csv

 


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
    env_type = os.environ.get("TREE_ENV_TYPE", "updown").lower()
    actions = make_actions(env_type)
    tokenizer = build_tree_tokenizer(actions)
    id_EOS = tokenizer.eos_token_id
    env = make_env(env_type, args.horizon)
    # Build action ids list in token order
    action_ids = make_action_ids(tokenizer, actions)
    logits_processor = TreeLogitsProcessor(env, tokenizer, action_ids, id_EOS)

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
    reward_fn = make_self_surprise_reward(trainer=None, tokenizer=tokenizer, env=env, action_ids=action_ids, id_eos=id_EOS)

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
    apply_masked_generate(trainer, logits_processor)

    # Replace reward function to dynamically use the current ref_model (π_k) after trainer init
    trainer.reward_funcs = [make_self_surprise_reward(trainer, tokenizer, env, action_ids, id_EOS)]
    trainer.reward_func_names = ["self_surprise_ref_k"]

    # Callback: log exact entropy over legal sequences at every step
    class ExactEntropyLogger(TrainerCallback):
        def __init__(self, trainer, tokenizer, env, action_ids: List[int], id_EOS: int, out_dir: str = "outputs"):
            self.trainer = trainer
            self.tokenizer = tokenizer
            self.env = env
            self.action_ids = action_ids
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
                H, finals = enumerate_sequence_probs(model_eval, self.tokenizer, self.env, self.action_ids, self.id_EOS)
            total_p = sum(p for _, p in finals)
            if total_p > 0:
                finals = [(a, p / total_p) for a, p in finals]
            # append CSV row only (single time-series file)
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([int(state.global_step), float(H)])

    trainer.add_callback(ExactEntropyLogger(trainer, tokenizer, env, action_ids, id_EOS))

    # Emit BEFORE distribution (initial policy) before training
    def dump_sequence_probs(model, csv_path):
        H, seqs = enumerate_sequence_probs(model, tokenizer, env, action_ids, id_EOS)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "probability"])
            for a, p in sorted(seqs, key=lambda x: -x[1]):
                w.writerow([a, p])
        return H, seqs

    before_csv = os.path.join(_PROJECT_ROOT, "outputs", "before_sequence_probs.csv")
    model_before = trainer.accelerator.unwrap_model(trainer.model)
    H_before, seqs_before = dump_sequence_probs(model_before, before_csv)
    legal_mass_before = compute_legal_mass_raw(model_before, tokenizer, env, action_ids, id_EOS)
    # Save exact entropy before finetuning
    with open(os.path.join(_PROJECT_ROOT, "outputs", "before_exact_entropy.json"), "w") as f:
        json.dump({"entropy_nats": float(H_before), "num_sequences": len(seqs_before)}, f, indent=2)

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
        seq_logp_ref = compute_sequence_logprobs(model_eval_ref, responses, attention_mask, tokenizer, env, action_ids, id_EOS)
        seq_logp_pol = compute_sequence_logprobs(model_eval, responses, attention_mask, tokenizer, env, action_ids, id_EOS)
        mean_reward = (-seq_logp_ref).mean().item()
        approx_kl = (seq_logp_pol - seq_logp_ref).mean().item()
        seq_entropy = (-seq_logp_pol).mean().item()

        # Per-action first-step probabilities under policy (masked, renormalized)
        # Determine allowed action ids at BOS using generic env API
        empty_prefix: List[int] = []
        if hasattr(env, "legal_action_ids"):
            allowed_first = env.legal_action_ids(empty_prefix, action_ids, tokenizer.eos_token_id)
        elif hasattr(env, "legal_actions_ids"):
            allowed_first = env.legal_actions_ids(empty_prefix, action_ids, tokenizer.eos_token_id)
        else:
            raise AttributeError("Environment must implement legal_action_ids or legal_actions_ids")
        # Compute next-token distribution at BOS
        bos = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
        logits0 = model_eval(input_ids=bos, attention_mask=torch.ones_like(bos)).logits[0, -1]
        logZ0 = torch.logsumexp(logits0[allowed_first], dim=-1)
        p_first = {aid: float(torch.exp(logits0[aid] - logZ0).item()) for aid in allowed_first}

        # Sample-based first action fractions
        first_actions = responses[:, 1]
        frac_actions = {aid: float((first_actions == aid).float().mean().item()) for aid in action_ids}
        has_eos = (responses == tokenizer.eos_token_id).any(dim=1).float().mean().item()

        # Prepare dynamic metrics dict
        metrics = {
            "eval/mean_reward": mean_reward,
            "eval/approx_kl": approx_kl,
            "eval/seq_entropy": seq_entropy,
            "eval/has_eos_frac": has_eos,
        }
        # Add per-action probabilities
        for a_tok, a_id in zip(actions, action_ids):
            metrics[f"eval/p_{a_tok}"] = p_first.get(a_id, 0.0)
            metrics[f"eval/frac_{a_tok}"] = frac_actions.get(a_id, 0.0)

        print(metrics)

    exact_H, seqs = enumerate_sequence_probs(model_eval, tokenizer, env, action_ids, id_EOS)
    legal_mass_after = compute_legal_mass_raw(model_eval, tokenizer, env, action_ids, id_EOS)
    cost_excess_invalid = max(0.0, float(legal_mass_before - legal_mass_after))
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
        json.dump(report | {"legal_mass_before_raw": float(legal_mass_before), "legal_mass_after_raw": float(legal_mass_after), "excess_invalid_mass": float(cost_excess_invalid)}, f, indent=2)
    with open(os.path.join(out_dir, "after_sequence_probs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "probability"])
        for a, p in sorted(seqs, key=lambda x: -x[1]):
            w.writerow([a, p])


if __name__ == "__main__":
    main()


