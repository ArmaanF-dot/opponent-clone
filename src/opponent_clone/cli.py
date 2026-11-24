"""CLI: opponent-clone fetch --lichess NAME --chesscom NAME [--max 500] [--out data]"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import chesscom, lichess
from .config import load_dotenv


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="opponent-clone")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download games and split by time control")
    f.add_argument("--lichess")
    f.add_argument("--chesscom")
    f.add_argument("--max", type=int, default=500, help="max games per platform")
    f.add_argument("--out", default="data")
    a = p.parse_args(argv)
    if not (a.lichess or a.chesscom):
        p.error("give --lichess and/or --chesscom")

    games = []
    if a.lichess:
        games += list(lichess.fetch(a.lichess, a.max))
    if a.chesscom:
        games += list(chesscom.fetch(a.chesscom, a.max))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    by_key: dict[str, list] = {}
    for g in games:
        by_key.setdefault(f"{g.source}_{g.bucket}", []).append(g.to_dict())
        counts[(g.source, g.bucket)] += 1
    for key, rows in by_key.items():
        (out / f"{key}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for (src, bkt), n in sorted(counts.items()):
        print(f"{src:8s} {bkt:10s} {n}")
    print(f"total {len(games)} games -> {out}/")


if __name__ == "__main__":
    main()
