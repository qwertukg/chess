# ------------------------------ Движок/состояние ---------------------------
import threading
from collections import deque
from typing import Dict, Optional, Tuple
import logging
import time

import chess
import torch

from neural.mini_az import MiniAZ
from mcts.mcts import MCTS
from self_play.training import play_self_game, train_on_batch
from self_play.sample import Sample
from utils import (
    DEVICE,
    DEFAULT_SIMS,
    REPLAY_CAP,
    WEIGHTS_FILE,
    normalize_fen,
    seed_everything,
    index_to_move,
)

log = logging.getLogger(__name__)


class Engine:
    def __init__(self) -> None:
        self.net = MiniAZ(channels=64, blocks=4).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.replay: deque[Sample] = deque(maxlen=REPLAY_CAP)
        self.board = chess.Board()
        self.player_is_white = True
        self.sims = DEFAULT_SIMS
        self.lock = threading.Lock()
        self.total_games = 0
        self.total_samples = 0
        self.total_time = 0.0
        self.total_loss = 0.0
        self.total_policy_loss = 0.0
        self.total_value_loss = 0.0
        self.total_sims = 0.0
        try:
            sd = torch.load(WEIGHTS_FILE, map_location=DEVICE)
            self.net.load_state_dict(sd)
            log.info(f"Weights loaded: {WEIGHTS_FILE}")
        except Exception as e:
            log.info(f"No weights yet (random net). {e}")

    def set_position(self, fen: str, player_color: str) -> Tuple[bool, str]:
        src = fen
        fen = normalize_fen(fen)
        log.info(f"set_position | src='{src}' -> norm='{fen}' | color={player_color}")
        try:
            b = chess.Board(fen)
        except Exception as e:
            log.warning(f"Bad FEN: {e}")
            return False, f"Неверный FEN: {e}"
        with self.lock:
            self.board = b
            self.player_is_white = player_color.lower().startswith('w')
        return True, "Позиция установлена."

    def human_move(self, uci_from: str, uci_to: str, promo: str = "q") -> Tuple[bool, str]:
        with self.lock:
            try:
                suffix = promo if self._needs_promo(uci_from, uci_to) else ""
                move = chess.Move.from_uci(uci_from + uci_to + suffix)
            except Exception:
                log.warning(f"human_move | bad uci: {uci_from}->{uci_to}+{promo}")
                return False, "Некорректный формат хода."
            if move not in self.board.legal_moves:
                log.info(f"human_move | illegal: {move.uci()}")
                return False, "Нелегальный ход."
            self.board.push(move)
            log.info(f"human_move | ok: {move.uci()} | fen='{self.board.fen()}'")
            return True, "Ход принят."

    def _needs_promo(self, frm: str, to: str) -> bool:
        a = chess.Move.from_uci(frm + to)
        piece = self.board.piece_at(a.from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return False
        rank_to = chess.square_rank(a.to_square)
        return (rank_to == 7 and self.board.turn == chess.WHITE) or (
            rank_to == 0 and self.board.turn == chess.BLACK
        )

    def ai_move(self) -> Tuple[bool, str, Optional[str]]:
        with self.lock:
            if self.board.is_game_over():
                log.info("ai_move | game over")
                return False, "Игра окончена.", None
            mcts = MCTS(self.net, sims=self.sims)
            a, _ = mcts.run(self.board, temperature=1e-6)
            if a == -1:
                log.info("ai_move | no moves")
                return False, "Нет ходов.", None
            frm, to, promo = index_to_move(a)
            move = chess.Move(frm, to, promotion=promo)
            if move not in self.board.legal_moves:
                log.warning("ai_move | mismatch with legal moves")
                return False, "Расхождение с легальными ходами.", None
            san = self.board.san(move)
            self.board.push(move)
            log.info(f"ai_move | {move.uci()} ({san}) | fen='{self.board.fen()}'")
            return True, f"ИИ сыграл: {san}", move.uci()

    def self_play_train(self, games: int, sims: int) -> Dict[str, float]:
        sims = int(max(16, sims))
        self.sims = sims
        t0 = time.time()
        seed = seed_everything()
        log.info(
            f"[train] start | seed={seed} games={games} sims={sims} "
            f"replay_cap={self.replay.maxlen} device={DEVICE}"
        )
        gpos = 0
        loss_sum = lp_sum = lv_sum = 0.0
        for g in range(1, games + 1):
            gt0 = time.time()
            samples = play_self_game(self.net, sims=sims)
            npos = len(samples)
            for s in samples:
                self.replay.append(s)
            gpos += npos
            log.info(f"[train] game {g}/{games} | positions={npos} | replay_size={len(self.replay)}")
            batch = list(self.replay)
            if len(batch) >= 2:
                loss, lp, lv = train_on_batch(self.net, self.opt, batch)
                loss_sum += loss
                lp_sum += lp
                lv_sum += lv
                log.info(
                    f"[train] optimize after game {g} | batch={len(batch)} | "
                    f"loss={loss:.4f} | policy={lp:.4f} | value={lv:.4f}"
                )
            else:
                log.info(f"[train] skip optimize (batch_size={len(batch)})")
            log.info(f"[train] game {g} done in {time.time() - gt0:.2f}s")
            try:
                torch.save(self.net.state_dict(), WEIGHTS_FILE)
                log.info(f"[train] saved weights after game {g} -> {WEIGHTS_FILE}")
            except Exception as e:
                log.error(f"[train] save after game {g} failed: {e}")
        dt = time.time() - t0
        n = max(1, games)
        avg_loss = loss_sum / n
        avg_lp = lp_sum / n
        avg_lv = lv_sum / n
        log.info(
            f"[train] done | games={games} samples={gpos} sims={sims} | "
            f"avg_loss={avg_loss:.4f} | avg_policy={avg_lp:.4f} | "
            f"avg_value={avg_lv:.4f} | time={dt:.2f}s"
        )
        self.total_games += games
        self.total_samples += gpos
        self.total_time += dt
        self.total_loss += avg_loss * games
        self.total_policy_loss += avg_lp * games
        self.total_value_loss += avg_lv * games
        self.total_sims += sims * games
        tot_games = max(1, self.total_games)
        return {
            "games": self.total_games,
            "samples": self.total_samples,
            "sims": self.total_sims / tot_games,
            "loss": self.total_loss / tot_games,
            "policy_loss": self.total_policy_loss / tot_games,
            "value_loss": self.total_value_loss / tot_games,
            "time_sec": self.total_time,
        }
