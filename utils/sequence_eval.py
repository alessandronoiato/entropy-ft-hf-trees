from typing import List, Tuple

import math
import torch


def compute_sequence_logprobs(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    tokenizer,
    env,
    action_ids: List[int],
    id_eos: int,
) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        next_logits = logits[:, :-1, :]
        next_ids = input_ids[:, 1:]
        B, S, _ = next_logits.shape
        device = next_logits.device
        token_logps = torch.zeros((B, S), dtype=next_logits.dtype, device=device)
        for b in range(B):
            seq_tokens = input_ids[b].tolist()
            for t in range(S):
                prefix_ids = seq_tokens[: t + 1]
                prefix_actions = [i for i in prefix_ids if i not in {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}]
                if hasattr(env, "legal_action_ids"):
                    allowed_ids = env.legal_action_ids(prefix_actions, action_ids, id_eos)
                elif hasattr(env, "legal_actions_ids"):
                    allowed_ids = env.legal_actions_ids(prefix_actions, action_ids, id_eos)
                else:
                    # Legacy: assume first two are U,D
                    allow_U, allow_D, allow_EOS = env.legal_actions(prefix_actions, action_ids[0], action_ids[1], id_eos)
                    allowed_ids = []
                    if allow_U:
                        allowed_ids.append(action_ids[0])
                    if allow_D:
                        allowed_ids.append(action_ids[1])
                    if allow_EOS:
                        allowed_ids.append(id_eos)
                if not allowed_ids:
                    continue
                logits_bt = next_logits[b, t]
                logZ = torch.logsumexp(logits_bt[allowed_ids], dim=-1)
                next_id = next_ids[b, t].item()
                logp = logits_bt[next_id] - logZ
                if attention_mask[b, t + 1].item() == 1:
                    token_logps[b, t] = logp
        return token_logps.sum(dim=-1)


def enumerate_sequence_probs(model, tokenizer, env, action_ids: List[int], id_eos: int) -> Tuple[float, List[Tuple[str, float]]]:
    device = next(model.parameters()).device
    bos = tokenizer.bos_token_id
    prefixes: List[Tuple[List[int], List[int], float]] = [([bos], [], 0.0)]
    finals: List[Tuple[str, float]] = []
    model.eval()
    with torch.no_grad():
        while prefixes:
            next_prefixes = []
            for token_ids, actions, logp_sum in prefixes:
                if hasattr(env, "legal_action_ids"):
                    allowed_ids = env.legal_action_ids(actions, action_ids, id_eos)
                elif hasattr(env, "legal_actions_ids"):
                    allowed_ids = env.legal_actions_ids(actions, action_ids, id_eos)
                else:
                    allow_U, allow_D, allow_EOS = env.legal_actions(actions, action_ids[0], action_ids[1], id_eos)
                    allowed_ids = []
                    if allow_U:
                        allowed_ids.append(action_ids[0])
                    if allow_D:
                        allowed_ids.append(action_ids[1])
                    if allow_EOS:
                        allowed_ids.append(id_eos)
                if not allowed_ids:
                    continue
                inp = torch.tensor([token_ids], dtype=torch.long, device=device)
                attn = torch.ones_like(inp)
                logits_last = model(input_ids=inp, attention_mask=attn).logits[0, -1]
                logZ = torch.logsumexp(logits_last[allowed_ids], dim=-1).item()
                for tok in allowed_ids:
                    logp_tok = (logits_last[tok].item() - logZ)
                    new_logp = logp_sum + logp_tok
                    if tok == id_eos:
                        actions_str = "".join([tokenizer.convert_ids_to_tokens([a])[0] for a in actions])
                        finals.append((actions_str, math.exp(new_logp)))
                    else:
                        next_prefixes.append((token_ids + [tok], actions + [tok], new_logp))
            prefixes = next_prefixes
    total_p = sum(p for _, p in finals)
    if total_p > 0:
        finals = [(a, p / total_p) for a, p in finals]
    H = -sum(p * math.log(p + 1e-40) for _, p in finals)
    return H, finals


def compute_legal_mass_raw(model, tokenizer, env, action_ids: List[int], id_eos: int) -> float:
    """Compute total raw probability mass assigned to legal sequences.

    Raw means we use the full softmax denominator over the entire vocabulary
    at each step (no legality renormalization). We still enumerate only legal
    transitions to build legal sequences, then sum their raw probabilities.
    """
    device = next(model.parameters()).device
    bos = tokenizer.bos_token_id
    prefixes: List[Tuple[List[int], List[int], float]] = [([bos], [], 0.0)]
    total_mass = 0.0
    model.eval()
    with torch.no_grad():
        while prefixes:
            next_prefixes = []
            for token_ids, actions, logp_sum in prefixes:
                if hasattr(env, "legal_action_ids"):
                    allowed_ids = env.legal_action_ids(actions, action_ids, id_eos)
                elif hasattr(env, "legal_actions_ids"):
                    allowed_ids = env.legal_actions_ids(actions, action_ids, id_eos)
                else:
                    raise AttributeError("Environment must implement legal_action_ids or legal_actions_ids")
                if not allowed_ids:
                    continue
                inp = torch.tensor([token_ids], dtype=torch.long, device=device)
                attn = torch.ones_like(inp)
                logits_last = model(input_ids=inp, attention_mask=attn).logits[0, -1]
                # Full-vocab normalization (raw mass)
                logZ_full = torch.logsumexp(logits_last, dim=-1).item()
                for tok in allowed_ids:
                    logp_tok = logits_last[tok].item() - logZ_full
                    new_logp = logp_sum + logp_tok
                    if tok == id_eos:
                        total_mass += math.exp(new_logp)
                    else:
                        next_prefixes.append((token_ids + [tok], actions + [tok], new_logp))
            prefixes = next_prefixes
    return float(total_mass)


