# ----------------------------- Нейросеть -----------------------------------
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.bn1(self.c1(x)))
        h = self.bn2(self.c2(h))
        return F.relu(x + h)
