"""Lichess bot that plays like the clone. Uses the Lichess Bot API (https://lichess.org/api#tag/Bot).

Security:
  * The token comes ONLY from the LICHESS_BOT_TOKEN environment variable (or a gitignored .env). It is never logged.
  * It declines every challenge unless the challenger is on LICHESS_BOT_ALLOW (comma-separated usernames).
  * It plays one game at a time and only unrated games.
Setup: create a NEW Lichess account for the bot, make a token with the 'bot:play' scope, then upgrade the account
to a bot account (this is permanent and only works on an account with no games played):
  curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer $LICHESS_BOT_TOKEN"
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable, Iterator

import chess
import requests

API = "https://lichess.org"


def board_from_state(initial_fen: str | None, moves: str) -> chess.Board:
    board = chess.Board() if not initial_fen or initial_fen == "startpos" else chess.Board(initial_fen)
    for uci in moves.split():
        board.push_uci(uci)
    return board


def my_clock(state: dict, white: bool) -> tuple[float | None, float | None]:
    """(seconds left, increment seconds) for our side, from a gameState event."""
    ms = state.get("wtime" if white else "btime")
    inc = state.get("winc" if white else "binc")
    return (ms / 1000 if isinstance(ms, (int, float)) else None, (inc or 0) / 1000)


def allowed(challenger: str, allow_env: str | None) -> bool:
    names = {n.strip().lower() for n in (allow_env or "").split(",") if n.strip()}
    return challenger.lower() in names


class Bot:
    def __init__(self, token: str, chooser: Callable, think: Callable, my_elo: int = 2500, allow: str | None = None):
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"
        self.chooser, self.think, self.my_elo = chooser, think, my_elo
        self.allow = allow if allow is not None else os.environ.get("LICHESS_BOT_ALLOW", "")
        self.busy = threading.Lock()

    def _stream(self, path: str) -> Iterator[dict]:
        with self.s.get(API + path, stream=True, timeout=(10, 120)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    yield json.loads(line)

    def run(self) -> None:
        me = self.s.get(f"{API}/api/account", timeout=20).json()
        self.me = me["id"]
        print(f"bot online as {me['username']}; accepting challenges from: {self.allow or 'nobody'}")
        for ev in self._stream("/api/stream/event"):
            if ev["type"] == "challenge":
                ch = ev["challenge"]
                ok = (allowed(ch["challenger"]["name"], self.allow) and not ch.get("rated")
                      and ch.get("variant", {}).get("key") == "standard" and not self.busy.locked())
                self.s.post(f"{API}/api/challenge/{ch['id']}/{'accept' if ok else 'decline'}", timeout=20)
            elif ev["type"] == "gameStart" and self.busy.acquire(blocking=False):
                threading.Thread(target=self._play, args=(ev["game"]["gameId"],), daemon=True).start()

    def _play(self, game_id: str) -> None:
        try:
            white, base, opp_elo, init = True, None, 2000, None
            for ev in self._stream(f"/api/bot/game/stream/{game_id}"):
                if ev["type"] == "gameFull":
                    white = ev["white"].get("id") == self.me
                    opp = ev["black" if white else "white"]
                    opp_elo = opp.get("rating") or 2000
                    clk = ev.get("clock") or {}
                    base = (clk.get("initial") or 0) / 1000 or None
                    init = ev.get("initialFen")
                    state = ev["state"]
                elif ev["type"] == "gameState":
                    state = ev
                else:
                    continue
                if state.get("status") not in ("started", "created"):
                    break
                board = board_from_state(init, state.get("moves", ""))
                if board.turn != white or board.is_game_over():
                    continue
                clock, _ = my_clock(state, white)
                uci = self.chooser(board, self.my_elo, opp_elo, clock, base)
                time.sleep(self.think(clock, base))
                self.s.post(f"{API}/api/bot/game/{game_id}/move/{uci}", timeout=20)
        finally:
            self.busy.release()


def main() -> None:
    import argparse

    from .clone import Clone
    from .config import load_dotenv

    load_dotenv()
    p = argparse.ArgumentParser(prog="opponent-clone-bot")
    p.add_argument("--head", required=True, help="clone_head.pt from the Kaggle run")
    p.add_argument("--clock", help="clock_model.json from the Kaggle run")
    p.add_argument("--elo", type=int, default=2500, help="rating the clone should play as")
    a = p.parse_args()
    token = os.environ.get("LICHESS_BOT_TOKEN", "").strip()
    if not token or token.startswith("your_"):
        raise SystemExit("set LICHESS_BOT_TOKEN in your environment or a local .env file (never commit it)")
    clone = Clone(a.head, a.clock)
    Bot(token, clone.choose, clone.think_time, a.elo).run()


if __name__ == "__main__":
    main()
