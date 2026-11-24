"""Turn fetched games into (position, move) training samples. Needs python-chess only (no torch)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import chess


def _base_inc(g: dict) -> tuple[float | None, float | None]:
    try:
        b, _, i = g["time_control"].partition("+")
        return float(b), float(i or 0)
    except (ValueError, KeyError):
        return None, None


def game_samples(g: dict, skip_plies: int = 0) -> list[dict]:
    """One sample per ply the *player* moved. fen is before the move; move is UCI."""
    board = chess.Board()
    mine_parity = 0 if g["color"] == "white" else 1
    out = []
    for i, san in enumerate(g["moves"].split()):
        try:
            mv = board.parse_san(san)
        except ValueError:
            break  # corrupt or non-standard game: keep what we have
        if i % 2 == mine_parity and i >= skip_plies and g.get("player_rating") and g.get("opponent_rating"):
            cl = g.get("clocks") or []
            base, inc = _base_inc(g)
            before = cl[i - 2] if i >= 2 and i - 2 < len(cl) else (base if i < 2 else None)
            after = cl[i] if i < len(cl) else None
            spent = max(0.0, before + inc - after) if None not in (before, after, inc) else None
            out.append({
                "game_id": f"{g['source']}:{g['game_id']}", "ply": i, "fen": board.fen(), "move": mv.uci(),
                "elo_self": g["player_rating"], "elo_oppo": g["opponent_rating"],
                "clock": before,   # own clock (s) before moving
                "base": base, "inc": inc, "spent": spent,   # seconds spent on this move
            })
        board.push(mv)
    return out


def split_games(games: list[dict], holdout_frac: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Chronological split by game: the newest games are held out, so there is no leakage from later play."""
    gs = sorted(games, key=lambda g: g["played_at"])
    k = max(1, int(len(gs) * holdout_frac)) if len(gs) >= 5 else 0
    return (gs[:-k], gs[-k:]) if k else (gs, [])


def build_dataset(jsonl_files: list[str], bucket: str, holdout_frac: float = 0.2, skip_plies: int = 0):
    games = []
    for f in jsonl_files:
        games += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    games = [g for g in games if g["bucket"] == bucket]
    tr, ho = split_games(games, holdout_frac)
    to = lambda gs: [s for g in gs for s in game_samples(g, skip_plies)]
    return to(tr), to(ho)
