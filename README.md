# Opponent Clone

Enter an opponent's usernames, pull their public games, split them by time control, and (later) build a style profile and a sparring bot that plays like them.

**Status:** build step 1 — fetch games from Lichess and Chess.com and split them into bullet / blitz / rapid / classical. Bucket = base + 40 × increment (<180s bullet, <480s blitz, <1500s rapid, else classical). Correspondence and variants are dropped.

## Use
```
pip install -e .[dev]
cp .env.example .env      # optional; .env is gitignored
opponent-clone fetch --lichess SOMEUSER --chesscom SOMEUSER --max 500
pytest
```
Output goes to `data/` (gitignored), one JSONL file per platform and time control, with clocks and openings.

## Security
- No keys in code. Settings come from environment variables or a local `.env`, which is gitignored. Only `.env.example` (placeholders) is committed.
- The public Lichess and Chess.com endpoints used here need no key. A Lichess token is optional and only raises rate limits; use one with no scopes.
- Downloaded games stay local (`data/`, `*.pgn` are gitignored).
