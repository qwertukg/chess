"""Пример запуска предобучения и последующего шага самообучения."""
import torch.optim as optim

from config import DEVICE
from neural.mini_az import MiniAZ
from pretrain.training import pretrain_on_openings
from self_play.training import play_self_game, train_on_batch


def main():
    net = MiniAZ().to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    pretrain_on_openings(net, opt)
    game = play_self_game(net)
    train_on_batch(net, opt, game)


if __name__ == "__main__":
    main()
