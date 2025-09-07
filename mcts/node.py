# ------------------------------ MCTS (PUCT) --------------------------------
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Node:
    prior: float
    to_play_white: bool
    parent: Optional["Node"] = None
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    children: Dict[int, "Node"] = None
    legal_idx: List[int] = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = {}
