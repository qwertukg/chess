# ------------------------ Самоигра + обучение ------------------------------
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import chess

from neural.mini_az import MiniAZ
from mcts.mcts import MCTS
from utils import (DEVICE, MAX_MOVES, TEMP_FIRST_MOVES, board_to_planes,
                   index_to_move)
from .sample import Sample


def play_self_game(net: MiniAZ, sims: int = 64,
                   temp_moves: int = TEMP_FIRST_MOVES,
                   max_moves: int = MAX_MOVES) -> List[Sample]:
    mcts = MCTS(net, sims=sims)
    board = chess.Board()
    hist: List[Sample] = []
    move_no = 0
    while not board.is_game_over(claim_draw=True) and move_no < max_moves:
        planes = board_to_planes(board)
        temperature = 1.0 if move_no < temp_moves else 1e-6
        a, visits = mcts.run(board, temperature=temperature)
        if a == -1:
            break
        legal_idx = mcts.policy_value(board)[2]
        counts = np.array([visits.get(i, 0) for i in legal_idx], dtype=np.float32)
        if counts.sum() == 0:
            counts += 1.0
        pi = counts / counts.sum()
        hist.append(Sample(planes, legal_idx, pi, z=0.0))
        frm, to, promo = index_to_move(a)
        board.push(chess.Move(frm, to, promotion=promo))
        move_no += 1

    if board.is_stalemate() or board.is_insufficient_material() or \
       board.is_fivefold_repetition() or board.is_seventyfive_moves():
        result = 0.0
    else:
        res = board.result()
        result = 0.0 if res == "1/2-1/2" else (1.0 if res == "1-0" else -1.0)

    turn_white = True
    for s in hist:
        s.z = result if turn_white else -result
        turn_white = not turn_white
    return hist


def train_on_batch(net: MiniAZ, opt, batch: List[Sample], l2: float = 1e-4) -> Tuple[float, float, float]:
    net.train()
    planes = torch.stack([s.planes for s in batch]).to(DEVICE)
    logits, values = net(planes)
    z = torch.tensor([s.z for s in batch], dtype=torch.float32, device=DEVICE)
    loss_v = F.mse_loss(values, z)
    loss_p = 0.0
    for i, s in enumerate(batch):
        idx = torch.tensor(s.legal_idx, dtype=torch.long, device=DEVICE)
        logp = F.log_softmax(logits[i, idx], dim=0)
        tgt = torch.tensor(s.pi, dtype=torch.float32, device=DEVICE)
        loss_p += -(tgt * logp).sum()
    loss_p = loss_p / max(1, len(batch))
    l2_reg = sum((p**2).sum() for n, p in net.named_parameters() if "bn" not in n)
    loss = loss_p + loss_v + l2 * l2_reg
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return float(loss.item()), float(loss_p.item()), float(loss_v.item())
