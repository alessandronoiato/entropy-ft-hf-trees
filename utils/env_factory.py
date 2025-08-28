from typing import List

from env.tree_env import UpDownTree, UpDownMiddleTree, TreeConfig


def make_actions(env_type: str) -> List[str]:
    et = (env_type or "updown").lower()
    return ["U", "D", "M"] if et in {"udm", "updownmiddle", "updown_middle"} else ["U", "D"]


def make_env(env_type: str, horizon: int):
    et = (env_type or "updown").lower()
    cfg = TreeConfig(horizon=horizon)
    if et in {"udm", "updownmiddle", "updown_middle"}:
        return UpDownMiddleTree(cfg)
    return UpDownTree(cfg)


def action_ids(tokenizer, actions: List[str]) -> List[int]:
    return [tokenizer.convert_tokens_to_ids([a])[0] for a in actions]


