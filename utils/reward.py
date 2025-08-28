from typing import List

import torch

from .sequence_eval import compute_sequence_logprobs


def make_self_surprise_reward(trainer, tokenizer, env, action_ids: List[int], id_eos: int):
    def reward_func(prompts, completions, completion_ids, **kwargs):
        ref = trainer.ref_model if trainer.ref_model is not None else trainer.model
        device = next(ref.parameters()).device
        prompt_ids_list = []
        for p in prompts:
            ids = tokenizer(text=p, return_tensors=None, add_special_tokens=False).get("input_ids", None)
            if ids is None:
                ids = tokenizer(p, add_special_tokens=False)["input_ids"]
            prompt_ids_list.append(ids)
        batch_ids = []
        for pid, cid in zip(prompt_ids_list, completion_ids):
            seq = torch.tensor(pid + cid, dtype=torch.long)
            batch_ids.append(seq)
        max_len = max(seq.size(0) for seq in batch_ids)
        input_ids = torch.full((len(batch_ids), max_len), fill_value=tokenizer.pad_token_id, dtype=torch.long)
        for i, seq in enumerate(batch_ids):
            input_ids[i, : seq.size(0)] = seq
        input_ids = input_ids.to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
        seq_logp_ref = compute_sequence_logprobs(ref, input_ids, attention_mask, tokenizer, env, action_ids, id_eos)
        return (-seq_logp_ref).float().cpu().tolist()

    return reward_func


