"""Chess.com public monthly archives. No key needed; send a descriptive User-Agent."""
from __future__ import annotations

import re
from typing import Iterator

import requests

from .config import user_agent
from .games import Game
from .timecontrol import bucket, parse_time_control

ARCHIVES = "https://api.chess.com/pub/player/{user}/games/archives"
CLOCK_RE = re.compile(r"\{\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]\}")
MOVE_NUM_RE = re.compile(r"\d+\.(\.\.)?")
RESULT_MAP = {"win": "win", "checkmated": "loss", "resigned": "loss", "timeout": "loss", "abandoned": "loss",
              "lose": "loss", "kingofthehill": "loss", "threecheck": "loss", "bughousepartnerlose": "loss"}


def _result(code: str) -> str:
    if code == "win":
        return "win"
    if code in ("agreed", "repetition", "stalemate", "insufficient", "50move", "timevsinsufficient"):
        return "draw"
    return "loss"


def _opening_from_pgn(pgn: str) -> tuple[str | None, str | None]:
    eco = re.search(r'\[ECO "([^"]+)"\]', pgn)
    url = re.search(r'\[ECOUrl "[^"]*/openings/([^"]+)"\]', pgn)
    name = None
    if url:
        # Drop trailing move list, e.g. "Sicilian-Defense-Open-2.Nf3-d6" -> "Sicilian Defense Open"
        name = re.split(r"-\d+\.|\.\.\.", url.group(1))[0].replace("-", " ").strip()
    return (name, eco.group(1) if eco else None)


def _moves_and_clocks(pgn: str) -> tuple[str, list[float]]:
    body = pgn.split("\n\n", 1)[-1]
    clocks = [int(h) * 3600 + int(m) * 60 + float(s) for h, m, s in CLOCK_RE.findall(body)]
    text = re.sub(r"\{[^}]*\}", " ", body)
    text = re.sub(r"\d+\.\.\.", " ", text)
    toks = [t for t in text.split() if not MOVE_NUM_RE.fullmatch(t) and not re.fullmatch(r"1-0|0-1|1/2-1/2|\*", t)]
    toks = [re.sub(r"^\d+\.", "", t) for t in toks]
    return " ".join(t for t in toks if t), clocks


def parse_game(raw: dict, user: str) -> Game | None:
    if raw.get("rules", "chess") != "chess" or not raw.get("rated", True):
        return None
    if raw.get("time_class") == "daily":
        return None
    tc = parse_time_control(raw.get("time_control"))
    if tc is None:
        return None
    w, b = raw["white"], raw["black"]
    if w["username"].lower() == user.lower():
        color, me, opp = "white", w, b
    elif b["username"].lower() == user.lower():
        color, me, opp = "black", b, w
    else:
        return None
    pgn = raw.get("pgn", "")
    name, eco = _opening_from_pgn(pgn)
    moves, clocks = _moves_and_clocks(pgn)
    return Game(
        source="chesscom", game_id=raw["url"].rstrip("/").split("/")[-1], player=user, color=color,
        opponent=opp["username"], player_rating=me.get("rating"), opponent_rating=opp.get("rating"),
        result=_result(me["result"]), bucket=bucket(*tc), time_control=raw["time_control"],
        played_at=int(raw.get("end_time", 0)) * 1000, opening=name, eco=eco, moves=moves, clocks=clocks, pgn=pgn,
    )


def fetch(user: str, max_games: int = 500, session: requests.Session | None = None) -> Iterator[Game]:
    s = session or requests.Session()
    s.headers["User-Agent"] = user_agent()
    r = s.get(ARCHIVES.format(user=user.lower()), timeout=30)
    r.raise_for_status()
    n = 0
    for url in reversed(r.json().get("archives", [])):  # newest month first
        m = s.get(url, timeout=60)
        m.raise_for_status()
        for raw in reversed(m.json().get("games", [])):
            g = parse_game(raw, user)
            if g:
                yield g
                n += 1
                if n >= max_games:
                    return
