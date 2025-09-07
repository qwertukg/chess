#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Минимальный AlphaZero-like движок с веб-GUI (исправленная версия):
# - Починено отображение доски после ввода FEN (chessboard.js принимает только 1-е поле FEN)
# - Добавлено логирование ключевых операций
# - Укреплена обработка ошибок, корректная нормализация FEN, мелкие правки MCTS (eval)
#
# Зависимости: Flask, python-chess, torch, numpy
# Установка:   pip install flask python-chess torch numpy
# Запуск:      python web_chess_az_fixed.py    -> http://127.0.0.1:5000

import math, random, json, time, threading, logging
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from flask import Flask, request, jsonify, Response

# ------------------------------ Логирование --------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("web_chess_az")

# ------------------------------ Конфигурация -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"); log.info(f"torch.device = {DEVICE}")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

WEIGHTS_FILE = "mini_az_web.pt"
REPLAY_CAP = 2048      # буфер повторов
DEFAULT_SIMS = 96      # число MCTS-симуляций на ход (можно менять в UI)
TEMP_FIRST_MOVES = 10  # сколько первых ходов использовать температуру=1.0
MAX_MOVES = 90         # лимит длины партии в self-play

# --------------------------- Представление позиции -------------------------
# 12 плоскостей фигур + 1 (чей ход) + 4 рокировки + 1 полуходы до ничьей
PLANES = 12 + 1 + 4 + 1
POLICY_SIZE = 64 * 64 * 5  # (from64 * to64 * промо {-,Q,R,B,N})

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

# ----------------------------- Нейросеть -----------------------------------
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)
    def forward(self, x):
        h = F.relu(self.bn1(self.c1(x)))
        h = self.bn2(self.c2(h))
        return F.relu(x + h)


class MiniAZ(nn.Module):
    def __init__(self, channels=64, blocks=4):
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

# ------------------------------ MCTS (PUCT) --------------------------------
@dataclass
class Node:
    prior: float
    to_play_white: bool
    parent: Optional["Node"] = None
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    children: Dict[int, "Node"] = None
    legal_idx: List[int] = None
    def __post_init__(self):
        if self.children is None:
            self.children = {}


def softmax_masked(logits: torch.Tensor, idx: List[int]) -> np.ndarray:
    with torch.no_grad():
        v = logits[idx]
        v = v - v.max()
        p = torch.exp(v); p = p / p.sum()
        return p.cpu().numpy()


class MCTS:
    def __init__(self, net: MiniAZ, sims=64, cpuct=1.2, dirichlet_eps=0.25, dir_alpha=0.3):
        self.net = net; self.sims = sims; self.cpuct = cpuct
        self.dir_eps = dirichlet_eps; self.dir_alpha = dir_alpha

    @torch.no_grad()
    def policy_value(self, board: chess.Board) -> Tuple[np.ndarray, float, List[int]]:
        self.net.eval()  # важно: BN/Dropout в инференсе
        x = board_to_planes(board).unsqueeze(0).to(DEVICE)
        logits, value = self.net(x)
        logits = logits[0]
        legal = list(board.legal_moves)
        legal_idx = [move_to_index(m) for m in legal]
        if len(legal_idx) == 0:
            return np.array([], dtype=np.float32), float(value.item()), legal_idx
        probs = softmax_masked(logits, legal_idx)
        return probs, float(value.item()), legal_idx

    def run(self, board: chess.Board, temperature=1.0) -> Tuple[int, Dict[int,int]]:
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
            self._backprop(node, v); return v

        if len(node.children) == 0:
            probs, v, legal_idx = self.policy_value(board)
            node.legal_idx = legal_idx
            for a, p in zip(legal_idx, probs):
                node.children[a] = Node(float(p), not node.to_play_white, parent=node)
            self._backprop(node, v); return v

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
            # редкая защита при рассинхронизации индексов
            best_child.N += 1
            return best_child.Q

        board.push(move)
        v = self._simulate(board, best_child)
        self._backprop(node, v)
        return v

    def _backprop(self, node: Node, v: float):
        cur = node
        while cur is not None:
            cur.N += 1
            cur.W += v
            cur.Q = cur.W / cur.N
            v = -v
            cur = cur.parent

# ------------------------ Самоигра + обучение ------------------------------
@dataclass
class Sample:
    planes: torch.Tensor
    legal_idx: List[int]
    pi: np.ndarray
    z: float


def play_self_game(net: MiniAZ, sims=64, temp_moves=10, max_moves=256) -> List[Sample]:
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
        # распределение из посещений
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

    # проставим цели value (взгляд игрока на ходе в каждой позиции)
    turn_white = True
    for s in hist:
        s.z = result if turn_white else -result
        turn_white = not turn_white
    return hist


def train_on_batch(net: MiniAZ, opt, batch: List[Sample], l2=1e-4) -> Tuple[float,float,float]:
    net.train()
    planes = torch.stack([s.planes for s in batch]).to(DEVICE)
    logits, values = net(planes)
    # value
    z = torch.tensor([s.z for s in batch], dtype=torch.float32, device=DEVICE)
    loss_v = F.mse_loss(values, z)
    # policy (только по легальным)
    loss_p = 0.0
    for i, s in enumerate(batch):
        idx = torch.tensor(s.legal_idx, dtype=torch.long, device=DEVICE)
        logp = F.log_softmax(logits[i, idx], dim=0)
        tgt = torch.tensor(s.pi, dtype=torch.float32, device=DEVICE)
        loss_p += -(tgt * logp).sum()
    loss_p = loss_p / max(1, len(batch))
    # L2 weight decay
    l2_reg = sum((p**2).sum() for n, p in net.named_parameters() if "bn" not in n)
    loss = loss_p + loss_v + l2 * l2_reg
    # шаг
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return float(loss.item()), float(loss_p.item()), float(loss_v.item())

# ------------------------------ Движок/состояние ---------------------------
class Engine:
    def __init__(self):
        self.net = MiniAZ(channels=64, blocks=4).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.replay: deque[Sample] = deque(maxlen=REPLAY_CAP)
        self.board = chess.Board()                 # текущая партия (веб-игра)
        self.player_is_white = True                # кто человек
        self.sims = DEFAULT_SIMS
        self.lock = threading.Lock()
        # попытка загрузить веса
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
            self.player_is_white = (player_color.lower().startswith('w'))
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
        # кто сейчас на ходу, тот и двигает
        return (rank_to == 7 and self.board.turn == chess.WHITE) or (rank_to == 0 and self.board.turn == chess.BLACK)

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
        log.info(f"[train] start | games={games} sims={sims} replay_cap={self.replay.maxlen} device={DEVICE}")

        gpos = 0
        loss_sum = lp_sum = lv_sum = 0.0

        for g in range(1, games + 1):
            gt0 = time.time()
            samples = play_self_game(self.net, sims=sims, temp_moves=TEMP_FIRST_MOVES, max_moves=MAX_MOVES)
            npos = len(samples)
            for s in samples:
                self.replay.append(s)
            gpos += npos

            log.info(f"[train] game {g}/{games} | positions={npos} | replay_size={len(self.replay)}")

            # простое обучение по всем накопленным
            batch = list(self.replay)
            if len(batch) >= 2:
                loss, lp, lv = train_on_batch(self.net, self.opt, batch)
                loss_sum += loss; lp_sum += lp; lv_sum += lv
                log.info(f"[train] optimize after game {g} | batch={len(batch)} | loss={loss:.4f} | policy={lp:.4f} | value={lv:.4f}")
            else:
                log.info(f"[train] skip optimize (batch_size={len(batch)})")

            log.info(f"[train] game {g} done in {time.time() - gt0:.2f}s")

        dt = time.time() - t0
        # сохраним веса
        try:
            torch.save(self.net.state_dict(), WEIGHTS_FILE)
            log.info(f"[train] saved weights -> {WEIGHTS_FILE}")
        except Exception as e:
            log.error(f"[train] save failed: {e}")

        n = max(1, games)
        avg_loss = loss_sum / n
        avg_lp = lp_sum / n
        avg_lv = lv_sum / n
        log.info(f"[train] done | games={games} samples={gpos} sims={sims} | avg_loss={avg_loss:.4f} | avg_policy={avg_lp:.4f} | avg_value={avg_lv:.4f} | time={dt:.2f}s")

        return {
            "games": games, "samples": gpos, "sims": sims,
            "loss": avg_loss, "policy_loss": avg_lp, "value_loss": avg_lv,
            "time_sec": dt
        }


engine = Engine()

# ------------------------------ Веб-сервер ---------------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.post("/init")
def api_init():
    data = request.get_json(force=True)
    fen = data.get("fen", chess.STARTING_FEN)
    color = data.get("color", "white")
    log.info(f"/init | fen='{fen}' | color={color}")
    ok, msg = engine.set_position(fen, color)
    if not ok:
        return jsonify({"ok": False, "message": msg})
    return jsonify({
        "ok": True,
        "message": msg,
        "fen": engine.board.fen(),
        "turn": "white" if engine.board.turn else "black"
    })


@app.post("/move")
def api_move():
    data = request.get_json(force=True)
    ufrom = data.get("from")
    uto = data.get("to")
    promo = data.get("promotion", "q")
    log.info(f"/move | {ufrom}->{uto} +{promo}")
    ok, msg = engine.human_move(ufrom, uto, promo)
    return jsonify({
        "ok": ok,
        "message": msg,
        "fen": engine.board.fen(),
        "game_over": engine.board.is_game_over()
    })


@app.post("/ai_move")
def api_ai():
    ok, msg, uci = engine.ai_move()
    return jsonify({
        "ok": ok,
        "message": msg,
        "fen": engine.board.fen(),
        "uci": uci,
        "game_over": engine.board.is_game_over()
    })


@app.post("/train")
def api_train():
    data = request.get_json(force=True)
    games = int(data.get("games", 2))
    sims = int(data.get("sims", DEFAULT_SIMS))
    log.info(f"/train | games={games} sims={sims}")
    stats = engine.self_play_train(games, sims)
    return jsonify({"ok": True, "stats": stats})


@app.get("/state")
def api_state():
    return jsonify({
        "fen": engine.board.fen(),
        "turn": "white" if engine.board.turn else "black"
    })

# ------------------------------- HTML/JS -----------------------------------
# ВАЖНО: chessboard.js ожидает только 1-е поле FEN (расстановку), без хода/рокировок и т.п.
# Поэтому на клиенте мы отрезаем всё после первого пробела перед board.position(...)
INDEX_HTML = r"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <title>Mini AlphaZero Chess (Web)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.css"/>
  <style>
    body{background:#202124;color:#eaeaea;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:0}
    .wrap{max-width:1100px;margin:24px auto;padding:0 16px;display:flex;gap:24px;align-items:flex-start}
    #board{width:560px;max-width:90vw}
    .panel{flex:1;background:#2b2d30;border:1px solid #3c4043;border-radius:12px;padding:16px}
    h1{font-size:20px;margin:0 0 12px}
    label{display:block;margin-top:8px;margin-bottom:6px;color:#cfcfcf}
    input[type=text],select,input[type=number]{width:100%;padding:8px 10px;border-radius:8px;border:1px solid #3c4043;background:#1f2124;color:#eaeaea}
    button{margin-top:10px;padding:10px 14px;border:1px solid #5865f2;background:#5865f2;color:#fff;border-radius:8px;cursor:pointer}
    button.secondary{background:#3c4043;border-color:#555}
    .row{display:flex;gap:8px;align-items:center}
    .msg{margin-top:10px;color:#ffd95e;min-height:24px}
    .small{opacity:0.8;font-size:12px}
    .kv{display:grid;grid-template-columns:140px 1fr;gap:6px 10px;margin-top:8px;font-size:14px}
    .kv div{padding:2px 0}
  </style>
</head>
<body>
<div class="wrap">
  <div id="board"></div>

  <div class="panel">
    <h1>Mini AlphaZero Chess (Web)</h1>

    <label>FEN (позиция)</label>
    <input id="fen" type="text" value="startpos"/>

    <div class="row">
      <div style="flex:1">
        <label>Сторона игрока</label>
        <select id="color">
          <option value="white" selected>Белые</option>
          <option value="black">Чёрные</option>
        </select>
      </div>
      <div style="flex:1">
        <label>Промоция пешки</label>
        <select id="promotion">
          <option value="q" selected>Ферзь (q)</option>
          <option value="r">Ладья (r)</option>
          <option value="b">Слон (b)</option>
          <option value="n">Конь (n)</option>
        </select>
      </div>
    </div>

    <div class="row">
      <button id="btnInit">Установить позицию</button>
      <button class="secondary" id="btnAI">Ход ИИ</button>
    </div>

    <label>Самообучение</label>
    <div class="row">
      <input id="games" type="number" min="1" max="50" value="2"/><span class="small">игр</span>
      <input id="sims" type="number" min="16" max="1024" value="96"/><span class="small">симуляций/ход</span>
      <button id="btnTrain">Тренировать</button>
    </div>

    <div class="kv small" id="stats"></div>
    <div class="msg" id="msg">Готово.</div>
    <div class="small">Подсказка: можно ввести только часть FEN (расстановка) — остальное заполнится по умолчанию.</div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js"></script>
<script>
let board = null;
let playerColor = 'white';
let promotion = 'q';

function toPieceFEN(fen){
  if(!fen) return 'start';
  const sp = fen.trim().split(/\s+/);
  return sp[0] || 'start';
}
function clearStorage(){
  try{
    localStorage.removeItem('az_stats');
    localStorage.removeItem('az_msg');
  }catch(e){/* ignore */}
}

function setMessage(s){
  document.getElementById('msg').textContent = s;
  try{ localStorage.setItem('az_msg', s); }catch(e){/* ignore */}
}

function setStats(obj){
  const el = document.getElementById('stats');
  if(!obj){ el.innerHTML = ''; return; }
  el.innerHTML = `
    <div>self-play игр</div><div>${obj.games}</div>
    <div>позиции</div><div>${obj.samples}</div>
    <div>симуляций/ход</div><div>${obj.sims}</div>
    <div>loss</div><div>${obj.loss.toFixed(4)}</div>
    <div>policy loss</div><div>${obj.policy_loss.toFixed(4)}</div>
    <div>value loss</div><div>${obj.value_loss.toFixed(4)}</div>
    <div>время, сек</div><div>${obj.time_sec.toFixed(1)}</div>
  `;
}

function initBoard(initialFen, orient){
  const cfg = {
    pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
    draggable: true,
    position: initialFen || 'start',
    orientation: orient || 'white',
    onDragStart: function (source, piece){
      // запретим таскать фигуры другого цвета
      if(playerColor === 'white' && piece[0] === 'b') return false;
      if(playerColor === 'black' && piece[0] === 'w') return false;
    },
    onDrop: async function (source, target){
      try{
        const res = await fetch('/move',{method:'POST',headers:{'Content-Type':'application/json'},
          body: JSON.stringify({from: source, to: target, promotion: promotion})});
        const j = await res.json();
        if(!j.ok){ setMessage(j.message || 'Нелегальный ход'); return 'snapback'; }
        board.position(toPieceFEN(j.fen));
        setMessage(j.message + (j.game_over ? ' (Игра окончена)' : ''));
      }catch(e){ setMessage('Ошибка соединения'); return 'snapback'; }
    }
  };
  const ChessCtor = window.Chessboard || window.ChessBoard;
  if(!ChessCtor){ console.error('Chessboard library is not loaded'); alert('Не удалось загрузить chessboard.js (CDN). Проверьте доступ к CDN или сеть.'); return; }
  board = ChessCtor('board', cfg);
}

async function refreshBoard(){
  const st = await fetch('/state').then(r=>r.json());
  board.position(toPieceFEN(st.fen));
}

document.getElementById('color').addEventListener('change', e=>{
  playerColor = e.target.value;
  board.orientation(playerColor);
});

document.getElementById('promotion').addEventListener('change', e=>{
  promotion = e.target.value;
});

document.getElementById('btnInit').addEventListener('click', async ()=>{
  let fen = document.getElementById('fen').value.trim();
  if(fen === '' || fen.toLowerCase()==='startpos'){ fen = 'startpos'; }
  if(fen === 'startpos'){ fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'; }
  const color = document.getElementById('color').value;
  const res = await fetch('/init',{method:'POST',headers:{'Content-Type':'application/json'},
    body: JSON.stringify({fen: fen, color: color})});
  const j = await res.json();
  if(!j.ok){ setMessage(j.message||'Ошибка'); return; }
  playerColor = color;
  board.orientation(color);
  board.position(toPieceFEN(j.fen));
  setMessage(j.message + ' Ход: ' + j.turn + '.');
});

document.getElementById('btnAI').addEventListener('click', async ()=>{
  const j = await fetch('/ai_move',{method:'POST'}).then(r=>r.json());
  if(!j.ok){ setMessage(j.message||'Ошибка'); return; }
  board.position(toPieceFEN(j.fen));
  setMessage(j.message + (j.game_over ? ' (Игра окончена)' : ''));
});

document.getElementById('btnTrain').addEventListener('click', async ()=>{
  clearStorage();
  setMessage('Тренировка запущена...');
  setStats(null);
  const games = Number(document.getElementById('games').value || 2);
  const sims  = Number(document.getElementById('sims').value || 96);
  const j = await fetch('/train',{method:'POST',headers:{'Content-Type':'application/json'},
    body: JSON.stringify({games: games, sims: sims})}).then(r=>r.json());
  if(!j.ok){ setMessage('Ошибка тренировки'); return; }
  setStats(j.stats);
  try{ localStorage.setItem('az_stats', JSON.stringify(j.stats)); }catch(e){/* ignore */}
  setMessage('Готово.');
});

window.addEventListener('load', async function boot(){
  playerColor = document.getElementById('color').value;
  promotion   = document.getElementById('promotion').value;
  try{
    const s = localStorage.getItem('az_stats');
    if(s){ setStats(JSON.parse(s)); }
  }catch(e){/* ignore */}
  let msg = null;
  try{ msg = localStorage.getItem('az_msg'); }catch(e){/* ignore */}
  if(msg){ setMessage(msg); }
  initBoard('start', playerColor);
  await refreshBoard();
});

const resetBtn = document.getElementById('btnReset');
if(resetBtn){ resetBtn.addEventListener('click', clearStorage); }
</script>
</body>
</html>
"""

# ------------------------------- main --------------------------------------
if __name__ == "__main__":
    log.info(f"Device: {DEVICE}")
    app.run(host="127.0.0.1", port=5000, debug=False)
