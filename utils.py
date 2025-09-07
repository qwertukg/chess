import math
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import chess
import torch

from config import PLANES

def seed_everything(seed: int | None = None) -> int:
    """Seed Python, NumPy and torch RNGs.

    If ``seed`` is ``None`` a seed derived from current time is used.
    The value actually used is returned so it can be logged.
    """
    if seed is None:
        seed = int(time.time()) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

# установить стартовое зерно один раз при запуске
seed_val = seed_everything()


def board_to_planes(b: chess.Board) -> torch.Tensor:
    planes = np.zeros((PLANES, 8, 8), dtype=np.float32)
    order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for i, piece in enumerate(order):
        for sq in b.pieces(piece, chess.WHITE):
            r, c = divmod(sq, 8); planes[i, 7 - r, c] = 1.0
        for sq in b.pieces(piece, chess.BLACK):
            r, c = divmod(sq, 8); planes[6 + i, 7 - r, c] = 1.0
    planes[12, :, :] = 1.0 if b.turn == chess.WHITE else 0.0
    planes[13, :, :] = 1.0 if b.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if b.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if b.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if b.has_queenside_castling_rights(chess.BLACK) else 0.0
    planes[17, :, :] = min(b.halfmove_clock / 50.0, 1.0)
    return torch.from_numpy(planes)


def normalize_fen(fen: str) -> str:
    """Дополняем FEN до 6 полей, спец-случай 'startpos'."""
    f = fen.strip()
    if f == "" or f.lower() == "startpos":
        return chess.STARTING_FEN
    parts = f.split()
    # пользователь мог ввести только расстановку
    while len(parts) < 6:
        if   len(parts) == 1: parts.append('w')
        elif len(parts) == 2: parts.append('-')
        elif len(parts) == 3: parts.append('-')
        elif len(parts) == 4: parts.append('0')
        elif len(parts) == 5: parts.append('1')
    norm = ' '.join(parts[:6])
    return norm

# ----------------------------- Кодировка ходов -----------------------------

def move_to_index(m: chess.Move) -> int:
    prom_map = {None:0, chess.QUEEN:1, chess.ROOK:2, chess.BISHOP:3, chess.KNIGHT:4}
    p = prom_map.get(m.promotion, 0)
    return (m.from_square * 64 + m.to_square) * 5 + p


def index_to_move(idx: int) -> Tuple[int,int,Optional[int]]:
    frm_to = idx // 5
    p = idx % 5
    frm, to = divmod(frm_to, 64)
    promo = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][p]
    return frm, to, promo
