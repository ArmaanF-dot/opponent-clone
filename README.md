# Opponent Clone

Enter an opponent's usernames, pull their public games, split them by time control, and (later) build a style profile and a sparring bot that plays like them.

**Status:** steps 1-2 done (fetch + rule-based style profile and prep report, no ML). Step 1 — fetch games from Lichess and Chess.com and split them into bullet / blitz / rapid / classical. Bucket = base + 40 × increment (<180s bullet, <480s blitz, <1500s rapid, else classical). Correspondence and variants are dropped.

## Use
```
pip install -e .[dev]
cp .env.example .env      # optional; .env is gitignored
opponent-clone fetch --lichess SOMEUSER --chesscom SOMEUSER --max 500
pytest
```
Then build the rule-based style profile and prep report:
```
opponent-clone report --player "Name" --data data --out report.md
```
Output goes to `data/` (gitignored), one JSONL file per platform and time control, with clocks and openings.

## Security
- No keys in code. Settings come from environment variables or a local `.env`, which is gitignored. Only `.env.example` (placeholders) is committed.
- The public Lichess and Chess.com endpoints used here need no key. A Lichess token is optional and only raises rate limits; use one with no scopes.
- Downloaded games stay local (`data/`, `*.pgn` are gitignored).

## Step 3: fine-tune a Maia-2 clone (Kaggle)
`kaggle/finetune_maia.py` (a one-cell Kaggle script that downloads the games itself) fine-tunes the pretrained Maia-2 blitz/rapid model on one player and one time control, then compares move-matching (top-1, top-3, loss) with base Maia-2 on that player's newest, held-out games. Set the usernames at the top, turn on a GPU and Internet in a new Kaggle notebook, paste the script into one cell and run (steps are at the top of the file; no keys needed). The default fine-tunes only the output head and rating embeddings; the clone must beat base on held-out games to count. Per-player and clock embeddings come in step 4.

Tested end to end on CPU with a small sample (pipeline works; use 100+ games for a real result).

## Step 4: clock-aware clone and Lichess bot
- The Kaggle script now also saves `clone_head.pt` (the fine-tuned layers only, about 8 MB) and `clock_model.json` (per-clock-bucket move temperature, so the clone gets sloppier short of time if the player does, plus the player's own think-time distribution). Download both from the notebook's Output tab. They stay out of git (`*.pt` is ignored).
- `opponent_clone.clone.Clone` loads base Maia-2 plus that head and picks moves; `pip install -e .[clone]`.
- `opponent-clone-bot --head clone_head.pt --clock clock_model.json` runs a Lichess bot. Use a NEW account made only for the bot, a token with only the `bot:play`, `challenge:read` and `challenge:write` scopes in `LICHESS_BOT_TOKEN` (env or gitignored `.env`), and list who may challenge it in `LICHESS_BOT_ALLOW` (it declines everyone otherwise, plays unrated standard games, one at a time). Upgrading an account to a bot is permanent: see the docstring in `bot.py`.
- Limits: clock awareness is a temperature plus think-time model, not a clock input to the network; the bot has not been tested against live Lichess yet (needs your bot account).

**Windows note:** `pip install -e .[clone]` can fail building `pyzstd` (a Maia-2 dependency the bot never uses). Use instead:
`pip install maia2 --no-deps`, `pip install torch chess einops gdown pandas pyyaml tqdm numpy requests`, `pip install -e . --no-deps`.
