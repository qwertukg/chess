# ----------------------------- Нейросеть -----------------------------------
import torch.nn as nn
import torch.nn.functional as F

from utils import PLANES, POLICY_SIZE
from .res_block import ResBlock


class MiniAZ(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(PLANES, channels, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(channels)
        self.res = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.head_p = nn.Sequential(
            nn.Conv2d(channels, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*8*8, 512), nn.ReLU(),
            nn.Linear(512, POLICY_SIZE)
        )
        self.head_v = nn.Sequential(
            nn.Conv2d(channels, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*8*8, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Tanh()
        )

    def forward(self, x):
        h = F.relu(self.bn0(self.stem(x)))
        h = self.res(h)
        p = self.head_p(h)
        v = self.head_v(h).squeeze(-1)
        return p, v
