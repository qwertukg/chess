#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Минимальный AlphaZero-like движок с веб-GUI

import logging

import chess
from flask import Flask, request, jsonify, Response

from engine.engine import Engine
from config import DEVICE
from utils import seed_val

# ------------------------------ Логирование --------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("web_chess_az")
log.info(f"torch.device = {DEVICE}")
log.info(f"Initial RNG seed = {seed_val}")

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
def api_ai_move():
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
    sims = int(data.get("sims", 96))
    log.info(f"/train | games={games} sims={sims}")
    stats = engine.self_play_train(games, sims)
    return jsonify({"ok": True, "stats": stats})


@app.get("/state")
def api_state():
    return jsonify({"fen": engine.board.fen()})

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
      <input id="games" type="number" min="1" max="50" value="2" placeholder="1:50"/><span class="small">игр</span>
      <input id="sims" type="number" min="16" max="1024" value="96" placeholder="16:1024"/><span class="small">симуляций/ход</span>
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
