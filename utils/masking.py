from typing import List

import torch
from transformers import LogitsProcessor


class TreeLogitsProcessor(LogitsProcessor):
    def __init__(self, env, tokenizer, id_U: int, id_D: int, id_EOS: int):
        super().__init__()
        self.env = env
        self.tok = tokenizer
        self.id_U = id_U
        self.id_D = id_D
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
            allow_U, allow_D, allow_EOS = self.env.legal_actions(actions, self.id_U, self.id_D, self.id_EOS)
            mask = torch.zeros(vocab, dtype=torch.bool, device=scores.device)
            # disallow everything by default
            mask[:] = True
            # allow subset
            if allow_U:
                mask[self.id_U] = False
            if allow_D:
                mask[self.id_D] = False
            if allow_EOS:
                mask[self.id_EOS] = False
            # Do not allow PAD to be generated; it's only for padding returned sequences
            scores[b][mask] = -1e9
        return scores


