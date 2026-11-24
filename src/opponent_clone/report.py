"""Markdown prep report, one section per time control."""
from __future__ import annotations

ORDER = ("bullet", "blitz", "rapid", "classical")


def pct(x, d=0):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def _tree_lines(tree: dict, indent="") -> list[str]:
    out = []
    for k, v in tree.items():
        if "share" in v:
            out.append(f"{indent}- {k}: {pct(v['share'])} ({v['n']})")
        else:
            out.append(f"{indent}- {k}")
            out += _tree_lines(v, indent + "  ")
    return out


def _flags(s: dict) -> list[str]:
    f = []
    if (s["time_trouble_pct"] or 0) > 0.35:
        f.append(f"Often short of time ({pct(s['time_trouble_pct'])} of games dip under 10% of the clock): keep the position complicated and let the clock work.")
    if (s["opening_time_used_pct"] or 0) > 0.2:
        f.append(f"Spends about {pct(s['opening_time_used_pct'])} of their time in the first 10 moves: surprise them early.")
    if s["score_after_loss"] is not None and s["n_after_loss"] >= 8 and s["score_vs_higher_rated"] is not None and s["score_after_loss"] < (s["score"] or 0) - 0.1:
        f.append(f"Scores {pct(s['score_after_loss'])} straight after a loss ({s['n_after_loss']} games) against {pct(s['score'])} overall: possible tilt.")
    if s["opposite_side_castling_pct"] > 0.15:
        f.append(f"Opposite-side castling in {pct(s['opposite_side_castling_pct'])} of games: expect pawn storms.")
    if s["short_games_pct"] > 0.25:
        f.append(f"{pct(s['short_games_pct'])} of games end before move 25: sharp, tactical player.")
    return f


def render(player: str, profile: dict) -> str:
    lines = [f"# Prep report: {player}", "",
             "Rule-based profile from public games. Style is not yet adjusted against same-rated players (needs a population baseline).", ""]
    for b in ORDER:
        if b not in profile:
            continue
        p, s = profile[b], profile[b]["stats"]
        lines += [f"## {b.capitalize()} ({s['games']} games; {', '.join(f'{k} {v}' for k, v in p['sources'].items())})", ""]
        if p["low_sample"]:
            lines += ["> Small sample: treat as indicative only.", ""]
        lines += [f"- Score {pct(s['score'])} (white {pct(s['score_white'])}, black {pct(s['score_black'])}); avg rating {s['avg_rating']:.0f}" if s["avg_rating"] else f"- Score {pct(s['score'])}",
                  f"- Median game length {s['median_game_length_moves']:.0f} moves; {pct(s['long_games_pct'])} reach move 50",
                  f"- Captures {pct(s['capture_rate'])} of own moves, checks {pct(s['check_rate'])}",
                  f"- Castling: short {pct(s['castle_short_pct'])}, long {pct(s['castle_long_pct'])}, opposite sides {pct(s['opposite_side_castling_pct'])}",
                  f"- Time trouble in {pct(s['time_trouble_pct'])} of games; first 10 moves use {pct(s['opening_time_used_pct'])} of the clock",
                  f"- After a loss: {pct(s['score_after_loss'])} ({s['n_after_loss']} games); vs higher-rated: {pct(s['score_vs_higher_rated'])} ({s['n_vs_higher_rated']}); vs lower-rated: {pct(s['score_vs_lower_rated'])} ({s['n_vs_lower_rated']})", ""]
        fl = _flags(s)
        if fl:
            lines += ["**Pointers**", ""] + [f"- {x}" for x in fl] + [""]
        lines += ["**As White, first move**", ""] + _tree_lines(p["white_first_moves"]) + [""]
        lines += ["**As Black, reply to each first move**", ""] + _tree_lines(p["black_replies"]) + [""]
        for col in ("white", "black"):
            lines += [f"**Top openings as {col.capitalize()}** (score)", ""]
            lines += [f"- {o['opening']}: {o["n"]} game(s), {pct(o['score'])}" for o in p[f"top_{col}_openings"]] + [""]
    return "\n".join(lines)
