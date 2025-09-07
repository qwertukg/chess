"""Парсинг PGN с дебютами в образцы для обучения."""
from pathlib import Path
from typing import List

import chess.pgn
import numpy as np

from utils import board_to_planes, move_to_index
from self_play.sample import Sample


def pgn_to_samples(path: Path) -> List[Sample]:
    """Читает PGN-файл и возвращает список образцов."""
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            board = game.board()
            hist: List[Sample] = []
            for move in game.mainline_moves():
                planes = board_to_planes(board)
                legal = list(board.legal_moves)
                legal_idx = [move_to_index(m) for m in legal]
                pi = np.zeros(len(legal_idx), dtype=np.float32)
                m_idx = move_to_index(move)
                try:
                    pos = legal_idx.index(m_idx)
                except ValueError:
                    board.push(move)
                    continue
                pi[pos] = 1.0
                hist.append(Sample(planes, legal_idx, pi, z=0.0))
                board.push(move)
            res = game.headers.get("Result", "1/2-1/2")
            if res == "1-0":
                result = 1.0
            elif res == "0-1":
                result = -1.0
            else:
                result = 0.0
            turn_white = True
            for s in hist:
                s.z = result if turn_white else -result
                turn_white = not turn_white
            samples.extend(hist)
    return samples


def load_opening_samples(directory: Path) -> List[Sample]:
    """Загружает все PGN-файлы из каталога."""
    all_samples: List[Sample] = []
    for pgn in sorted(directory.glob("*.pgn")):
        all_samples.extend(pgn_to_samples(pgn))
    return all_samples
