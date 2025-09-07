# ------------------------ Самоигра + обучение ------------------------------
from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class Sample:
    planes: torch.Tensor
    legal_idx: List[int]
    pi: np.ndarray
    z: float
