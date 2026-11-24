# Fine-tune Maia-2 on one player and one time control, then compare with the base model on held-out games.
# ONE-CELL KAGGLE RUN: it installs the repo, downloads the games itself, trains, and prints a verdict.
#
# Steps:
#  1. kaggle.com -> Create -> New Notebook. Right panel: Session options -> Accelerator "GPU T4 x2", Internet "On".
#  2. Paste this whole file into the first cell, edit LICHESS_USER / CHESSCOM_USER below, press Run.
# No Kaggle or Lichess keys are needed or used. Never put tokens in this file or the repo.

# %% setup
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "maia2", "chess", "requests",
                "git+https://github.com/ArmaanF-dot/opponent-clone.git"], check=True)

LICHESS_USER = "EricRosen"      # set to None to skip
CHESSCOM_USER = "IMRosen"       # set to None to skip
MAX_GAMES = 1500                # per platform; more games = better clone (aim for 100+ in your chosen bucket)

import json, random, chess, torch, torch.nn.functional as F
from opponent_clone import chesscom, lichess
from opponent_clone.samples import split_games, game_samples
from opponent_clone import clockmodel

CONFIG = dict(
    bucket="blitz",                        # blitz or rapid (Maia-2 ships these two base models)
    epochs=5, lr=1e-4, batch_size=64,
    train_scope="head",                    # "head" = output layer + final norm + elo embeddings; "all" = whole network
    holdout_frac=0.2, skip_plies=0, seed=0,
    out="clone_head.pt",                   # small file: only the fine-tuned layers (a few MB)
    clock_out="clock_model.json",
    max_positions=None,                    # cap per split for quick CPU smoke tests
)
if CONFIG["bucket"] not in ("blitz", "rapid"):
    sys.exit("Maia-2 only ships blitz and rapid base models; bullet/classical need your own base. Use blitz or rapid.")

games = []
if LICHESS_USER:
    games += [g.to_dict() for g in lichess.fetch(LICHESS_USER, MAX_GAMES)]
if CHESSCOM_USER:
    games += [g.to_dict() for g in chesscom.fetch(CHESSCOM_USER, MAX_GAMES)]
games = [g for g in games if g["bucket"] == CONFIG["bucket"]]
print(f"fetched {len(games)} {CONFIG['bucket']} games")

# maia2 imports pyzstd (only used for its own training-data decompression); its compiled module can fail to
# load on Kaggle, so stub it if it does. We never call it.
try:
    import pyzstd  # noqa: F401
except Exception:
    import types
    sys.modules["pyzstd"] = types.ModuleType("pyzstd")

from maia2 import model as maia_model
from maia2.utils import get_all_possible_moves, create_elo_dict, board_to_tensor, map_to_category, mirror_move

random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
all_moves = get_all_possible_moves()
move_to_idx = {m: i for i, m in enumerate(all_moves)} if isinstance(all_moves, list) else dict(all_moves)
elo_dict = create_elo_dict()

# %% data
tr_g, ho_g = split_games(games, CONFIG["holdout_frac"])
train_s = [x for g in tr_g for x in game_samples(g, CONFIG["skip_plies"])]
hold_s = [x for g in ho_g for x in game_samples(g, CONFIG["skip_plies"])]
print(f"train positions {len(train_s)}, held-out positions {len(hold_s)}")
if CONFIG["max_positions"]:
    train_s, hold_s = train_s[:CONFIG["max_positions"]], hold_s[:CONFIG["max_positions"]]
assert train_s and hold_s, "need data in both splits: raise MAX_GAMES or use a more active player (aim for 100+ games)"

def encode(batch):
    boards, es, eo, legal, y = [], [], [], [], []
    for s in batch:
        b = chess.Board(s["fen"]); mv = s["move"]
        if b.turn == chess.BLACK:          # Maia-2 always sees the side to move as White
            b = b.mirror(); mv = mirror_move(mv)
        boards.append(board_to_tensor(b))
        es.append(map_to_category(s["elo_self"], elo_dict)); eo.append(map_to_category(s["elo_oppo"], elo_dict))
        lm = torch.zeros(len(move_to_idx))
        for m in b.legal_moves: lm[move_to_idx[m.uci()]] = 1
        legal.append(lm); y.append(move_to_idx[mv])
    return (torch.stack(boards).to(device), torch.tensor(es).to(device), torch.tensor(eo).to(device),
            torch.stack(legal).to(device), torch.tensor(y).to(device))

def masked_logits(model, boards, es, eo, legal):
    logits, _, _ = model(boards, es, eo)
    return logits.masked_fill(legal == 0, float("-inf"))

@torch.no_grad()
def evaluate(model, samples):
    model.eval(); hit1 = hit3 = nll = 0.0
    for i in range(0, len(samples), 128):
        boards, es, eo, legal, y = encode(samples[i:i + 128])
        lg = masked_logits(model, boards, es, eo, legal)
        top = lg.topk(3, dim=-1).indices
        hit1 += (top[:, 0] == y).sum().item(); hit3 += (top == y[:, None]).any(1).sum().item()
        nll += F.cross_entropy(lg, y, reduction="sum").item()
    n = len(samples)
    return dict(top1=hit1 / n, top3=hit3 / n, nll=nll / n)

# %% base model baseline
base_type = CONFIG["bucket"]
model = maia_model.from_pretrained(type=base_type, device=device)
base = evaluate(model, hold_s)
print("base Maia-2  :", base)

# %% fine-tune
if CONFIG["train_scope"] == "head":
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(("fc_1.", "last_ln.", "elo_embedding."))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CONFIG["lr"], weight_decay=0.01)
best, best_state = None, None
for ep in range(CONFIG["epochs"]):
    model.train(); random.shuffle(train_s); tot = 0.0
    for i in range(0, len(train_s), CONFIG["batch_size"]):
        boards, es, eo, legal, y = encode(train_s[i:i + CONFIG["batch_size"]])
        loss = F.cross_entropy(masked_logits(model, boards, es, eo, legal), y)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(y)
    m = evaluate(model, hold_s)
    print(f"epoch {ep + 1}: train nll {tot / len(train_s):.3f} | held-out {m}")
    if best is None or m["nll"] < best["nll"]:      # early-stopping on held-out loss
        best, best_state = m, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

# %% verdict
print("\nbase Maia-2   :", base)
print("best clone    :", best)
print("PASS: clone beats base on held-out move matching" if best["top1"] > base["top1"] else
      "FAIL: clone does not beat base; need more games or a lower learning rate")

# %% save slim weights (only layers that were trained; the base model is downloaded again when you use the clone)
HEAD_PREFIXES = ("fc_1.", "last_ln.", "elo_embedding.") if CONFIG["train_scope"] == "head" else ("",)
slim = {k: v for k, v in best_state.items() if k.startswith(HEAD_PREFIXES)}
torch.save({"state": slim, "bucket": CONFIG["bucket"], "train_scope": CONFIG["train_scope"],
            "players": {"lichess": LICHESS_USER, "chesscom": CHESSCOM_USER}, "held_out": best, "base_held_out": base},
           CONFIG["out"])
print("saved", CONFIG["out"], f"({sum(v.numel() for v in slim.values()) * 4 / 1e6:.1f} MB)")

# %% clock model: per-clock-bucket temperature (does the player get sloppier short of time?) + think times
model.load_state_dict(best_state); model.eval()
per_bucket = {}
with torch.no_grad():
    for i in range(0, len(hold_s), 128):
        chunk = hold_s[i:i + 128]
        boards, es, eo, legal, y = encode(chunk)
        lg = masked_logits(model, boards, es, eo, legal).cpu()
        for j, smp in enumerate(chunk):
            idx = legal[j].nonzero().flatten().cpu()
            if smp["clock"] is None or smp["base"] is None:
                continue
            pos = (idx == y[j].cpu()).nonzero()
            if len(pos) == 0:
                continue
            b = clockmodel.clock_bucket(smp["clock"], smp["base"])
            per_bucket.setdefault(b, []).append((lg[j][idx].numpy(), int(pos[0])))
cm_model = clockmodel.fit(train_s + hold_s, per_bucket)
clockmodel.save(cm_model, CONFIG["clock_out"])
print("clock model: temperature by bucket (low clock -> high clock):", cm_model["temperature"])
print("samples per bucket:", cm_model["n_per_bucket"])
print("median think time (s) by bucket:", [sorted(t)[len(t) // 2] if t else None for t in cm_model["think_times"]])
print(f"\nDOWNLOAD: {CONFIG['out']} and {CONFIG['clock_out']} from the notebook's Output tab (/kaggle/working).")
