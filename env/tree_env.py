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
            if t + 1 >= self.config.horizon:
                return True, False, True
            else:
                return True, False, False
        # if first == D -> both allowed until EOS at horizon
        if first == id_D:
            if t + 1 >= self.config.horizon:
                return True, True, True
            else:
                return True, True, False
        # fallback: before first U/D, both allowed
        return True, True, False


