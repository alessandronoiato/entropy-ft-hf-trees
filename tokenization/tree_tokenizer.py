from typing import List, Optional, Dict

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


TREE_TOKENS: Dict[str, int] = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "U": 3,
    "D": 4,
}


def build_tree_tokenizer() -> PreTrainedTokenizerFast:
    # Build a tiny word-level tokenizer with a fixed vocab
    model = WordLevel(vocab=TREE_TOKENS, unk_token=None)
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
    tokenizer.bos_token_id = TREE_TOKENS["<bos>"]
    tokenizer.eos_token_id = TREE_TOKENS["<eos>"]
    tokenizer.pad_token_id = TREE_TOKENS["<pad>"]

    return tokenizer


def encode_actions(tokenizer: PreTrainedTokenizerFast, actions: List[str], add_bos: bool = True, add_eos: bool = False) -> List[int]:
    tokens = ([] if not add_bos else [tokenizer.bos_token]) + actions + ([] if not add_eos else [tokenizer.eos_token])
    return tokenizer.convert_tokens_to_ids(tokens)


def decode_actions(tokenizer: PreTrainedTokenizerFast, ids: List[int]) -> List[str]:
    tokens = tokenizer.convert_ids_to_tokens(ids)
    # strip special tokens
    return [t for t in tokens if t not in {tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token}]

