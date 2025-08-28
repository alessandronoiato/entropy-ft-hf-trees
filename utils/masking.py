from typing import List

import torch
from transformers import LogitsProcessor


class TreeLogitsProcessor(LogitsProcessor):
    def __init__(self, env, tokenizer, action_ids: List[int], id_EOS: int):
        super().__init__()
        self.env = env
        self.tok = tokenizer
        self.action_ids = action_ids
        self.id_EOS = id_EOS

    def _ids_to_actions(self, ids: List[int]) -> List[int]:
        # strip BOS if present and specials
        actions = [i for i in ids if i not in {self.tok.bos_token_id, self.tok.eos_token_id, self.tok.pad_token_id}]
        return actions

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch, vocab = scores.shape
        assert input_ids.shape[0] == batch
        for b in range(batch):
            ids = input_ids[b].tolist()
            actions = self._ids_to_actions(ids)
            # Ask env for allowed ids, supporting both 2-action and 3-action trees (generic API only)
            if hasattr(self.env, "legal_action_ids"):
                allowed = self.env.legal_action_ids(actions, self.action_ids, self.id_EOS)
            elif hasattr(self.env, "legal_actions_ids"):
                allowed = self.env.legal_actions_ids(actions, self.action_ids, self.id_EOS)
            else:
                raise AttributeError("Environment must implement legal_action_ids or legal_actions_ids")
            mask = torch.ones(vocab, dtype=torch.bool, device=scores.device)
            for a in allowed:
                mask[a] = False
            # Do not allow PAD to be generated; it's only for padding returned sequences
            scores[b][mask] = -1e9
        return scores



