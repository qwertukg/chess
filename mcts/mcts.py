# ------------------------------ MCTS (PUCT) --------------------------------
import math
from typing import Dict, List, Tuple

import numpy as np
import chess
import torch

from neural.mini_az import MiniAZ
from utils import board_to_planes, move_to_index, index_to_move
from .node import Node


def softmax_masked(logits: torch.Tensor, idx: List[int]) -> np.ndarray:
    with torch.no_grad():
        v = logits[idx]
        v = v - v.max()
        p = torch.exp(v)
        p = p / p.sum()
        return p.cpu().numpy()


class MCTS:
    def __init__(self, net: MiniAZ, sims: int = 64, cpuct: float = 1.2,
                 dirichlet_eps: float = 0.25, dir_alpha: float = 0.3) -> None:
        self.net = net
        self.sims = sims
        self.cpuct = cpuct
        self.dir_eps = dirichlet_eps
        self.dir_alpha = dir_alpha

    @torch.no_grad()
    def policy_value(self, board: chess.Board) -> Tuple[np.ndarray, float, List[int]]:
        self.net.eval()
        x = board_to_planes(board).unsqueeze(0).to(self.net.head_p[-1].weight.device)
        logits, value = self.net(x)
        logits = logits[0]
        legal = list(board.legal_moves)
        legal_idx = [move_to_index(m) for m in legal]
        if len(legal_idx) == 0:
            return np.array([], dtype=np.float32), float(value.item()), legal_idx
        probs = softmax_masked(logits, legal_idx)
        return probs, float(value.item()), legal_idx

    def run(self, board: chess.Board, temperature: float = 1.0) -> Tuple[int, Dict[int, int]]:
        root = Node(1.0, board.turn == chess.WHITE)
        probs, v, legal_idx = self.policy_value(board)
        root.legal_idx = legal_idx
        if len(probs) > 0:
            noise = np.random.dirichlet([self.dir_alpha] * len(probs))
            probs = (1 - self.dir_eps) * probs + self.dir_eps * noise
            for a, p in zip(legal_idx, probs):
                root.children[a] = Node(float(p), not root.to_play_white, parent=root)

        for _ in range(self.sims):
            self._simulate(board.copy(stack=False), root)

        visits = {a: ch.N for a, ch in root.children.items()}
        if not visits:
            return -1, {}
        if temperature <= 1e-6:
            best_a = max(visits.items(), key=lambda kv: kv[1])[0]
        else:
            acts, counts = zip(*visits.items())
            pi = np.array(counts, dtype=np.float32) ** (1.0 / temperature)
            pi = pi / pi.sum()
            best_a = int(np.random.choice(acts, p=pi))
        return best_a, visits

    def _simulate(self, board: chess.Board, node: Node) -> float:
        if board.is_game_over():
            res = board.result()
            v = 0.0 if res == "1/2-1/2" else (1.0 if (res == "1-0") == node.to_play_white else -1.0)
            self._backprop(node, v)
            return v

        if len(node.children) == 0:
            probs, v, legal_idx = self.policy_value(board)
            node.legal_idx = legal_idx
            for a, p in zip(legal_idx, probs):
                node.children[a] = Node(float(p), not node.to_play_white, parent=node)
            self._backprop(node, v)
            return v

        best, best_a, best_child = -1e9, None, None
        sqrtN = math.sqrt(node.N + 1)
        for a, ch in node.children.items():
            U = self.cpuct * ch.prior * (sqrtN / (1 + ch.N))
            score = ch.Q + U
            if score > best:
                best, best_a, best_child = score, a, ch

        frm, to, promo = index_to_move(best_a)
        move = chess.Move(frm, to, promotion=promo)
        if move not in board.legal_moves:
            best_child.N += 1
            return best_child.Q

        board.push(move)
        v = self._simulate(board, best_child)
        self._backprop(node, v)
        return v

    def _backprop(self, node: Node, v: float) -> None:
        cur = node
        while cur is not None:
            cur.N += 1
            cur.W += v
            cur.Q = cur.W / cur.N
            v = -v
            cur = cur.parent
