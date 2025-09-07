"""Предобучение сети на дебютных партиях."""
from pathlib import Path
from random import shuffle
from typing import List

from neural.mini_az import MiniAZ
from self_play.training import train_on_batch
from .openings import load_opening_samples


def pretrain_on_openings(net: MiniAZ, opt, data_dir: str = "data/openings", batch_size: int = 64):
    """Запускает один проход обучения на дебютных позициях."""
    samples = load_opening_samples(Path(data_dir))
    if not samples:
        return []
    shuffle(samples)
    losses: List[float] = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        loss, *_ = train_on_batch(net, opt, batch)
        losses.append(loss)
    return losses
