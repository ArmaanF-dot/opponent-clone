"""Lichess games export (NDJSON). Public games need no token; a token only raises rate limits."""
from __future__ import annotations

import json
from typing import Iterator

import requests

from .config import lichess_token, user_agent
from .games import Game
from .timecontrol import bucket

API = "https://lichess.org/api/games/user/{user}"
PERF_BUCKETS = {"ultraBullet": "bullet", "bullet": "bullet", "blitz": "blitz", "rapid": "rapid", "classical": "classical"}


def parse_game(raw: dict, user: str) -> Game | None:
    """Convert one Lichess export object; None for variants, correspondence, or unfinished games."""
    if raw.get("variant", "standard") != "standard":
        return None
    speed = raw.get("speed")
    if speed == "correspondence" or raw.get("status") in ("created", "started", "aborted", "noStart"):
        return None
    players = raw.get("players", {})
    w, b = players.get("white", {}), players.get("black", {})
    wname = (w.get("user") or {}).get("name", "")
    bname = (b.get("user") or {}).get("name", "")
    if wname.lower() == user.lower():
        color, me, opp, oname = "white", w, b, bname
    elif bname.lower() == user.lower():
        color, me, opp, oname = "black", b, w, wname
    else:
        return None
    clock = raw.get("clock")
    if clock:
        bkt = bucket(clock["initial"], clock["increment"])
        tc = f"{clock['initial']}+{clock['increment']}"
    else:
        bkt = PERF_BUCKETS.get(speed or "")
        tc = "?"
    if bkt is None:
        return None
    winner = raw.get("winner")
    result = "draw" if winner is None else ("win" if winner == color else "loss")
    op = raw.get("opening") or {}
    cs = raw.get("clocks") or []  # centiseconds, one per ply
    return Game(
        source="lichess", game_id=raw["id"], player=user, color=color, opponent=oname or "anonymous",
        player_rating=me.get("rating"), opponent_rating=opp.get("rating"), result=result,
        bucket=bkt, time_control=tc, played_at=raw.get("createdAt", 0),
        opening=op.get("name"), eco=op.get("eco"), moves=raw.get("moves", ""),
        clocks=[c / 100 for c in cs],
    )


def fetch(user: str, max_games: int = 500, since_ms: int | None = None, session: requests.Session | None = None) -> Iterator[Game]:
    headers = {"Accept": "application/x-ndjson", "User-Agent": user_agent()}
    tok = lichess_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    params = {"max": max_games, "clocks": "true", "opening": "true", "moves": "true",
              "pgnInJson": "false", "rated": "true", "finished": "true"}
    if since_ms:
        params["since"] = since_ms
    s = session or requests.Session()
    with s.get(API.format(user=user), headers=headers, params=params, stream=True, timeout=60) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            g = parse_game(json.loads(line), user)
            if g:
                yield g
