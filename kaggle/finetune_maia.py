# Fine-tune Maia-2 on one player and one time control, then compare with the base model on held-out games.
# Runs on Kaggle (free GPU) or locally on CPU for tiny data. No API keys or accounts are needed or used here.
#
# Kaggle steps:
#  1. New notebook -> Settings -> Accelerator: GPU T4 x2 (or P100). Internet: On.
#  2. Add your fetched data (data/*.jsonl from `opponent-clone fetch`) as a *private* Kaggle dataset.
#     It is public game data, but keep it private anyway.
#  3. Paste this file into a cell (or `!git clone` the repo) and set the CONFIG below.
# Never put Kaggle/Lichess tokens in this file or the repo.

# %% setup
# !pip install -q maia2 chess
import glob, json, random, sys
import chess, torch, torch.nn.functional as F

CONFIG = dict(
    data_glob="/kaggle/input/*/*.jsonl",   # where your dataset's jsonl files are
    bucket="blitz",                        # blitz or rapid (Maia-2 ships these two base models)
    epochs=5, lr=1e-4, batch_size=64,
    train_scope="head",                    # "head" = output layer + final norm + elo embeddings; "all" = whole network
    holdout_frac=0.2, skip_plies=0, seed=0,
    out="/kaggle/working/clone.pt",
    max_positions=None,                    # cap per split for quick CPU smoke tests
)
if CONFIG["bucket"] not in ("blitz", "rapid"):
    sys.exit("Maia-2 only ships blitz and rapid base models; bullet/classical need your own base. Use blitz or rapid.")

# repo helper (pure python). If you didn't clone the repo, copy src/opponent_clone/samples.py next to this file.
try:
    from opponent_clone.samples import build_dataset
except ImportError:
    from samples import build_dataset

from maia2 import model as maia_model
from maia2.utils import get_all_possible_moves, create_elo_dict, board_to_tensor, map_to_category, mirror_move

random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
all_moves = get_all_possible_moves()
move_to_idx = {m: i for i, m in enumerate(all_moves)} if isinstance(all_moves, list) else dict(all_moves)
elo_dict = create_elo_dict()

# %% data
train_s, hold_s = build_dataset(sorted(glob.glob(CONFIG["data_glob"])), CONFIG["bucket"], CONFIG["holdout_frac"], CONFIG["skip_plies"])
print(f"train positions {len(train_s)}, held-out positions {len(hold_s)}")
if CONFIG["max_positions"]:
    train_s, hold_s = train_s[:CONFIG["max_positions"]], hold_s[:CONFIG["max_positions"]]
assert train_s and hold_s, "need data in both splits: fetch more games (aim for 100+)"

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
torch.save(best_state, CONFIG["out"]); print("saved", CONFIG["out"])
