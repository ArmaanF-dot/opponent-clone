"""Rule-based style profile per (time-control bucket). No ML, no engine: derived from SAN moves, clocks and results."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

BOOK_PLIES = 16  # opening tree depth (8 moves each side)


def load_games(data_dir: str) -> list[dict]:
    rows = []
    for f in sorted(Path(data_dir).glob("*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def _score(games: list[dict]) -> float | None:
    if not games:
        return None
    pts = sum(1 if g["result"] == "win" else 0.5 if g["result"] == "draw" else 0 for g in games)
    return pts / len(games)


def _plies(g: dict) -> list[str]:
    return g["moves"].split() if g["moves"] else []


def opening_tree(games: list[dict], color: str, depth: int = 6, min_n: int = 2) -> dict:
    """First-move distribution as the player's own moves (white: ply 1; black: reply to 1.e4 etc.)."""
    mine = [g for g in games if g["color"] == color]
    tree: dict = {}
    if color == "white":
        c = Counter(_plies(g)[0] for g in mine if _plies(g))
        total = sum(c.values())
        tree = {m: {"n": n, "share": n / total} for m, n in c.most_common() if n >= min_n}
    else:
        by_first: dict[str, Counter] = defaultdict(Counter)
        for g in mine:
            p = _plies(g)
            if len(p) >= 2:
                by_first[p[0]][p[1]] += 1
        for first, c in by_first.items():
            total = sum(c.values())
            tree[f"vs 1.{first}"] = {m: {"n": n, "share": n / total} for m, n in c.most_common() if n >= min_n}
    return tree


def top_openings(games: list[dict], color: str, k: int = 6) -> list[dict]:
    mine = [g for g in games if g["color"] == color and g.get("opening")]
    by: dict[str, list] = defaultdict(list)
    for g in mine:
        by[g["opening"]].append(g)
    out = [{"opening": o, "n": len(gs), "score": _score(gs)} for o, gs in by.items()]
    return sorted(out, key=lambda x: -x["n"])[:k]


def _base_seconds(g: dict) -> float | None:
    try:
        return float(g["time_control"].split("+")[0])
    except (ValueError, KeyError):
        return None


def style_stats(games: list[dict]) -> dict:
    n = len(games)
    plies = [_plies(g) for g in games]
    own = [p[0 if g["color"] == "white" else 1::2] for g, p in zip(games, plies)]
    own_total = sum(len(o) for o in own) or 1
    captures = sum(1 for o in own for m in o if "x" in m)
    checks = sum(1 for o in own for m in o if m.endswith(("+", "#")))
    long_c = sum(1 for g, p in zip(games, plies) if any(m.startswith("O-O-O") for m in p[0 if g["color"] == "white" else 1::2]))
    short_c = sum(1 for g, p in zip(games, plies) if any(m.startswith("O-O") and not m.startswith("O-O-O") for m in p[0 if g["color"] == "white" else 1::2]))
    opp_side = 0
    for g, p in zip(games, plies):
        mine_c, theirs_c = set(), set()
        for i, m in enumerate(p):
            if m.startswith("O-O"):
                (mine_c if (i % 2 == 0) == (g["color"] == "white") else theirs_c).add("long" if m.startswith("O-O-O") else "short")
        if mine_c and theirs_c and mine_c != theirs_c:
            opp_side += 1
    lengths = [len(p) / 2 for p in plies]
    wins = [g for g in games if g["result"] == "win"]
    loss = [g for g in games if g["result"] == "loss"]

    # time usage from clocks (seconds remaining after each ply)
    trouble, early_spend, timed = 0, [], 0
    lost_on_low_clock = 0
    for g in games:
        base = _base_seconds(g)
        cl = g.get("clocks") or []
        if not base or len(cl) < 4:
            continue
        mine_cl = cl[0 if g["color"] == "white" else 1::2]
        if not mine_cl:
            continue
        timed += 1
        if min(mine_cl) < 0.1 * base:
            trouble += 1
            if g["result"] == "loss":
                lost_on_low_clock += 1
        if len(mine_cl) >= 10:
            early_spend.append((base - mine_cl[9]) / base)

    # after-loss performance (chronological)
    chron = sorted(games, key=lambda g: g["played_at"])
    after_loss = [b for a, b in zip(chron, chron[1:]) if a["result"] == "loss" and b["played_at"] - a["played_at"] < 3600_000]
    # vs rating gap
    def gap(g):
        return None if g["player_rating"] is None or g["opponent_rating"] is None else g["opponent_rating"] - g["player_rating"]
    higher = [g for g in games if (gap(g) or 0) > 50]
    lower = [g for g in games if (gap(g) or 0) < -50]
    return {
        "games": n,
        "score": _score(games),
        "score_white": _score([g for g in games if g["color"] == "white"]),
        "score_black": _score([g for g in games if g["color"] == "black"]),
        "avg_rating": mean([g["player_rating"] for g in games if g["player_rating"]]) if any(g["player_rating"] for g in games) else None,
        "median_game_length_moves": median(lengths) if lengths else None,
        "capture_rate": captures / own_total,
        "check_rate": checks / own_total,
        "castle_short_pct": short_c / n,
        "castle_long_pct": long_c / n,
        "opposite_side_castling_pct": opp_side / n,
        "short_games_pct": sum(1 for l in lengths if l < 25) / n,
        "long_games_pct": sum(1 for l in lengths if l >= 50) / n,
        "time_trouble_pct": trouble / timed if timed else None,
        "losses_from_time_trouble_share": lost_on_low_clock / len(loss) if loss and timed else None,
        "opening_time_used_pct": mean(early_spend) if early_spend else None,
        "score_after_loss": _score(after_loss),
        "n_after_loss": len(after_loss),
        "score_vs_higher_rated": _score(higher),
        "n_vs_higher_rated": len(higher),
        "score_vs_lower_rated": _score(lower),
        "n_vs_lower_rated": len(lower),
        "wins_by_short_game_pct": sum(1 for g in wins if len(_plies(g)) / 2 < 25) / len(wins) if wins else None,
    }


def build_profile(games: list[dict]) -> dict:
    """{bucket: {...}} with one profile per time control; sources kept separate only via the 'source' counts."""
    by: dict[str, list] = defaultdict(list)
    for g in games:
        by[g["bucket"]].append(g)
    prof = {}
    for b, gs in by.items():
        prof[b] = {
            "stats": style_stats(gs),
            "sources": dict(Counter(g["source"] for g in gs)),
            "white_first_moves": opening_tree(gs, "white"),
            "black_replies": opening_tree(gs, "black"),
            "top_white_openings": top_openings(gs, "white"),
            "top_black_openings": top_openings(gs, "black"),
            "low_sample": len(gs) < 30,
        }
    return prof
