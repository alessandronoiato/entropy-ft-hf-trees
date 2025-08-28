from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TreeConfig:
    horizon: int = 6


class UpDownTree:
    """
    Up/Down tree with horizon H.
    - From root, both U and D are allowed.
    - If first action is U, subsequent steps are forced to U (a line).
    - If first action is D, subsequent steps allow both U and D (branching line).
    """

    def __init__(self, config: TreeConfig):
        self.config = config

    def legal_actions(self, prefix: List[int], id_U: int, id_D: int, id_EOS: int) -> Tuple[bool, bool, bool]:
        # returns (allow_U, allow_D, allow_EOS)
        t = len(prefix)
        if t == 0:
            return True, True, False
        if t >= self.config.horizon:
            return False, False, True
        first = prefix[0]
        # if first == U -> only U allowed thereafter until EOS at horizon
        if first == id_U:
            # Only U allowed; EOS only when t >= horizon
            if t >= self.config.horizon:
                return False, False, True
            return True, False, False
        # if first == D -> both allowed until EOS at horizon
        if first == id_D:
            # U and D allowed; EOS only when t >= horizon
            if t >= self.config.horizon:
                return False, False, True
            return True, True, False
        # fallback: before first U/D, both allowed
        return True, True, False

    # Generic helper returning allowed token ids for current prefix
    def legal_action_ids(self, prefix: List[int], action_ids: List[int], eos_id: int) -> List[int]:
        # action_ids is expected to be [id_U, id_D] for this env
        id_U, id_D = action_ids[0], action_ids[1]
        allow_U, allow_D, allow_EOS = self.legal_actions(prefix, id_U, id_D, eos_id)
        allowed: List[int] = []
        if allow_U:
            allowed.append(id_U)
        if allow_D:
            allowed.append(id_D)
        if allow_EOS:
            allowed.append(eos_id)
        return allowed


class UpDownMiddleTree:
    """
    Up/Down/Middle tree with horizon H.
    - From any node before horizon, U, D, M are allowed; EOS only when t >= horizon.
    """

    def __init__(self, config: TreeConfig):
        self.config = config

    # For compatibility with existing processor when needed
    def legal_actions_ids(self, prefix: List[int], action_ids: List[int], eos_id: int) -> List[int]:
        t = len(prefix)
        allowed: List[int] = []
        if t >= self.config.horizon:
            return [eos_id]
        # allow all provided action ids before horizon
        allowed.extend(action_ids)
        # EOS only when t >= horizon (handled above), so do not append here
        return allowed



