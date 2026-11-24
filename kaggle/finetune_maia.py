# Fine-tune Maia-2 on one player and one time control: tries several training recipes, picks the best on a
# validation split, and reports the final score on a separate untouched test split (newest games).
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
MAX_GAMES = 4000                # per platform; more games = better clone (aim for 100+ in your chosen bucket)

import json, random, time, chess, torch, torch.nn.functional as F
from opponent_clone import chesscom, lichess
from opponent_clone.samples import split_games, game_samples
from opponent_clone import clockmodel

BUCKET = "blitz"                # blitz or rapid (Maia-2 ships these two base models)
BATCH = 128
OUT, CLOCK_OUT = "clone_head.pt", "clock_model.json"
SKIP_PLIES = 0
MAX_POSITIONS = None            # cap per split for quick CPU smoke tests

# Training recipes to compare. lr is per parameter group. recency_days = half-life of game weight (None = equal).
#   scope "head": output layer + final norm + rating embeddings (small file)
#   scope "top" : the above + both transformer layers (about 50 MB)
#   scope "all" : the whole network (about 90 MB)
EXPERIMENTS = [
    dict(name="head",            scope="head", lr_head=1e-4, lr_body=0,    epochs=4, recency_days=None),
    dict(name="top",             scope="top",  lr_head=1e-4, lr_body=2e-5, epochs=4, recency_days=None),
    dict(name="all",             scope="all",  lr_head=1e-4, lr_body=1e-5, epochs=4, recency_days=None),
    dict(name="all+recency",     scope="all",  lr_head=1e-4, lr_body=1e-5, epochs=4, recency_days=180),
]
HEAD_PREFIXES = ("fc_1.", "last_ln.", "elo_embedding.")
TOP_PREFIXES = HEAD_PREFIXES + ("transformer.", "pos_embedding", "to_patch_embedding.")
SEED = 0

if BUCKET not in ("blitz", "rapid"):
    sys.exit("Maia-2 only ships blitz and rapid base models; bullet/classical need your own base. Use blitz or rapid.")

games = []
if LICHESS_USER:
    games += [g.to_dict() for g in lichess.fetch(LICHESS_USER, MAX_GAMES)]
if CHESSCOM_USER:
    games += [g.to_dict() for g in chesscom.fetch(CHESSCOM_USER, MAX_GAMES)]
games = [g for g in games if g["bucket"] == BUCKET]
print(f"fetched {len(games)} {BUCKET} games")

# maia2 imports pyzstd (only used for its own training-data decompression); its compiled module can fail to
# load on Kaggle, so stub it if it does. We never call it.
try:
    import pyzstd  # noqa: F401
except Exception:
    import types
    sys.modules["pyzstd"] = types.ModuleType("pyzstd")

from maia2 import model as maia_model
from maia2.utils import get_all_possible_moves, create_elo_dict, board_to_tensor, map_to_category, mirror_move

random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
all_moves = get_all_possible_moves()
move_to_idx = {m: i for i, m in enumerate(all_moves)} if isinstance(all_moves, list) else dict(all_moves)
elo_dict = create_elo_dict()

# %% data: chronological split by game. train (oldest 70%) / validation (next 15%, picks the recipe and epoch,
# fits the clock model) / test (newest 15%, touched once at the end, so the reported score is honest)
rest_g, test_g = split_games(games, 0.15)
train_g, val_g = split_games(rest_g, 0.15 / 0.85)
mk = lambda gs: [x for g in gs for x in game_samples(g, SKIP_PLIES)]
train_s, val_s, test_s = mk(train_g), mk(val_g), mk(test_g)
if MAX_POSITIONS:
    train_s, val_s, test_s = train_s[:MAX_POSITIONS], val_s[:MAX_POSITIONS], test_s[:MAX_POSITIONS]
print(f"positions: train {len(train_s)}, validation {len(val_s)}, test {len(test_s)}")
assert train_s and val_s and test_s, "need data in all splits: raise MAX_GAMES or use a more active player (aim for 100+ games)"

def encode(batch):
    boards, es, eo, legal, y = [], [], [], [], []
    for s in batch:
        b = chess.Board(s["fen"]); mv = s["move"]
        if b.turn == chess.BLACK:          # Maia-2 always sees the side to move as White
            b = b.mirror(); mv = mirror_move(mv)
        boards.append(board_to_tensor(b))
        es.append(map_to_category(s["elo_self"], elo_dict)); eo.append(map_to_category(s["elo_oppo"], elo_dict))
        lm = torch.zeros(len(move_to_idx), dtype=torch.bool)
        for m in b.legal_moves: lm[move_to_idx[m.uci()]] = True
        legal.append(lm); y.append(move_to_idx[mv])
    return (torch.stack(boards).half(), torch.tensor(es), torch.tensor(eo), torch.stack(legal), torch.tensor(y))

def encode_all(samples, chunk=2048):
    parts = [encode(samples[i:i + chunk]) for i in range(0, len(samples), chunk)]
    return tuple(torch.cat([p[k] for p in parts]) for k in range(5))   # encode once, reuse for every recipe

t0 = time.time()
TR, VA, TE = encode_all(train_s), encode_all(val_s), encode_all(test_s)
print(f"encoded positions in {time.time() - t0:.0f}s")

def to_dev(T, idx):
    b, es, eo, legal, y = (t[idx] for t in T)
    return b.float().to(device), es.to(device), eo.to(device), legal.to(device), y.to(device)

def masked_logits(model, boards, es, eo, legal):
    logits, _, _ = model(boards, es, eo)
    return logits.masked_fill(~legal, float("-inf"))

@torch.no_grad()
def evaluate(model, T):
    model.eval(); hit1 = hit3 = nll = 0.0; n = len(T[0])
    for i in range(0, n, 256):
        boards, es, eo, legal, y = to_dev(T, slice(i, i + 256))
        lg = masked_logits(model, boards, es, eo, legal)
        top = lg.topk(3, dim=-1).indices
        hit1 += (top[:, 0] == y).sum().item(); hit3 += (top == y[:, None]).any(1).sum().item()
        nll += F.cross_entropy(lg, y, reduction="sum").item()
    return dict(top1=hit1 / n, top3=hit3 / n, nll=nll / n)

def sample_weights(samples, recency_days):
    if not recency_days:
        return torch.ones(len(samples))
    gt = {g["game_id"]: g["played_at"] for g in train_g}
    ts = torch.tensor([gt.get(s["game_id"].split(":", 1)[1], 0) for s in samples], dtype=torch.float)
    age_days = (ts.max() - ts) / 86.4e6
    w = 0.5 ** (age_days / recency_days)
    return w / w.mean()

# %% base model baseline (validation and test)
base_model = maia_model.from_pretrained(type=BUCKET, device=device)
base_val, base_test = evaluate(base_model, VA), evaluate(base_model, TE)
print("base Maia-2 validation:", base_val)
del base_model

# %% train every recipe, select on validation
def run(exp):
    model = maia_model.from_pretrained(type=BUCKET, device=device)
    for n, p in model.named_parameters():
        in_head = n.startswith(HEAD_PREFIXES)
        p.requires_grad = in_head if exp["scope"] == "head" else (n.startswith(TOP_PREFIXES) if exp["scope"] == "top" else True)
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(HEAD_PREFIXES)]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(HEAD_PREFIXES)]
    groups = [dict(params=head, lr=exp["lr_head"])] + ([dict(params=body, lr=exp["lr_body"])] if body else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    w_all = sample_weights(train_s, exp["recency_days"])
    best, best_state = None, None
    for ep in range(exp["epochs"]):
        model.train(); perm = torch.randperm(len(TR[0])); tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            boards, es, eo, legal, y = to_dev(TR, idx)
            ce = F.cross_entropy(masked_logits(model, boards, es, eo, legal), y, reduction="none")
            loss = (ce * w_all[idx].to(device)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += ce.sum().item()
        m = evaluate(model, VA)
        print(f"  {exp['name']} epoch {ep + 1}: train nll {tot / len(perm):.3f} | validation {m}")
        if best is None or m["nll"] < best["nll"]:
            best, best_state = dict(m, epoch=ep + 1), {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best, best_state

results = []
for exp in EXPERIMENTS:
    t0 = time.time(); print(f"\n== {exp['name']} ==")
    best, state = run(exp)
    results.append((best, state, exp)); print(f"  best epoch {best['epoch']} in {time.time() - t0:.0f}s")

# %% verdict: pick the recipe with the best validation loss, then score it ONCE on the untouched test split
print("\nvalidation results (top-1 / top-3 / loss):")
print(f"  base Maia-2      {base_val['top1']:.4f} / {base_val['top3']:.4f} / {base_val['nll']:.4f}")
for b, _, e in results:
    print(f"  {e['name']:<16} {b['top1']:.4f} / {b['top3']:.4f} / {b['nll']:.4f}  (epoch {b['epoch']})")
best, best_state, best_exp = min(results, key=lambda r: r[0]["nll"])
model = maia_model.from_pretrained(type=BUCKET, device=device)
model.load_state_dict(best_state)
final = evaluate(model, TE)
print(f"\nchosen recipe: {best_exp['name']}")
print("TEST  base Maia-2 :", base_test)
print("TEST  clone       :", final)
print(f"PASS: clone beats base on the held-out test games (top-1 {base_test['top1']:.3f} -> {final['top1']:.3f})"
      if final["top1"] > base_test["top1"] else "FAIL: clone does not beat base on the test games")

# %% save only the layers that were trained (the base model is downloaded again when you use the clone)
prefixes = {"head": HEAD_PREFIXES, "top": TOP_PREFIXES, "all": ("",)}[best_exp["scope"]]
slim = {k: v for k, v in best_state.items() if k.startswith(prefixes)}
torch.save({"state": slim, "bucket": BUCKET, "recipe": best_exp,
            "players": {"lichess": LICHESS_USER, "chesscom": CHESSCOM_USER},
            "test": final, "base_test": base_test}, OUT)
print("saved", OUT, f"({sum(v.numel() for v in slim.values()) * 4 / 1e6:.1f} MB)")

# %% clock model: per-clock-bucket temperature (does the player get sloppier short of time?) + think times
# fitted on the validation positions with the chosen model
model.eval(); per_bucket = {}
with torch.no_grad():
    for i in range(0, len(val_s), 256):
        boards, es, eo, legal, y = to_dev(VA, slice(i, i + 256))
        lg = masked_logits(model, boards, es, eo, legal).cpu()
        for j in range(len(y)):
            smp = val_s[i + j]
            if smp["clock"] is None or smp["base"] is None:
                continue
            idx = legal[j].nonzero().flatten().cpu()
            pos = (idx == y[j].cpu()).nonzero()
            if len(pos) == 0:
                continue
            b = clockmodel.clock_bucket(smp["clock"], smp["base"])
            per_bucket.setdefault(b, []).append((lg[j][idx].numpy(), int(pos[0])))
cm_model = clockmodel.fit(train_s + val_s + test_s, per_bucket)
clockmodel.save(cm_model, CLOCK_OUT)
print("clock model: temperature by bucket (low clock -> high clock):", cm_model["temperature"])
print("samples per bucket:", cm_model["n_per_bucket"])
print(f"\nDOWNLOAD: {OUT} and {CLOCK_OUT} from the notebook's Output tab (/kaggle/working).")
