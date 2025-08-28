from typing import List, Optional, Dict

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


BASE_TOKENS: Dict[str, int] = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
}


def build_tree_tokenizer(actions: Optional[List[str]] = None) -> PreTrainedTokenizerFast:
    # Build a tiny word-level tokenizer with a fixed vocab
    if actions is None:
        actions = ["U", "D"]
    vocab = dict(BASE_TOKENS)
    next_id = max(vocab.values()) + 1
    for a in actions:
        if a not in vocab:
            vocab[a] = next_id
            next_id += 1
    model = WordLevel(vocab=vocab, unk_token=None)
    tok = Tokenizer(model)
    tok.pre_tokenizer = Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    # map token ids explicitly
    tokenizer.add_tokens([], special_tokens=True)
    tokenizer.bos_token_id = vocab["<bos>"]
    tokenizer.eos_token_id = vocab["<eos>"]
    tokenizer.pad_token_id = vocab["<pad>"]

    return tokenizer


def encode_actions(tokenizer: PreTrainedTokenizerFast, actions: List[str], add_bos: bool = True, add_eos: bool = False) -> List[int]:
    tokens = ([] if not add_bos else [tokenizer.bos_token]) + actions + ([] if not add_eos else [tokenizer.eos_token])
    return tokenizer.convert_tokens_to_ids(tokens)


def decode_actions(tokenizer: PreTrainedTokenizerFast, ids: List[int]) -> List[str]:
    tokens = tokenizer.convert_ids_to_tokens(ids)
    # strip special tokens
    return [t for t in tokens if t not in {tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token}]



