"""Load a fine-tuned clone (base Maia-2 + the small saved head) and pick moves like the player would.

Needs the optional deps:  pip install maia2 torch chess
"""
from __future__ import annotations

import random

import chess

from . import clockmodel
from ._compat import stub_pyzstd


class Clone:
    def __init__(self, head_path: str, clock_path: str | None = None, device: str = "cpu", seed: int | None = None):
        import torch
        stub_pyzstd()
        from maia2 import model as maia_model
        from maia2.utils import create_elo_dict, get_all_possible_moves

        self.torch = torch
        self.rng = random.Random(seed)
        self.device = device
        ck = torch.load(head_path, map_location="cpu")
        self.meta = {k: v for k, v in ck.items() if k != "state"}
        self.model = maia_model.from_pretrained(type=ck["bucket"], device=device)
        self.model.load_state_dict(ck["state"], strict=False)  # only the fine-tuned layers are replaced
        self.model.eval()
        self.moves = get_all_possible_moves()
        self.elo_dict = create_elo_dict()
        self.clock = clockmodel.load(clock_path) if clock_path else None

    def move_probs(self, board: chess.Board, elo_self: int, elo_oppo: int, clock: float | None = None,
                   base: float | None = None) -> dict[str, float]:
        """Probability of each legal move (UCI), temperature-scaled by how much time is left."""
        from maia2.utils import board_to_tensor, map_to_category, mirror_move

        torch = self.torch
        flip = board.turn == chess.BLACK
        b = board.mirror() if flip else board
        idx_of = {m: i for i, m in enumerate(self.moves)}
        legal = [m.uci() for m in b.legal_moves]
        x = board_to_tensor(b).unsqueeze(0).to(self.device)
        es = torch.tensor([map_to_category(elo_self, self.elo_dict)]).to(self.device)
        eo = torch.tensor([map_to_category(elo_oppo, self.elo_dict)]).to(self.device)
        with torch.no_grad():
            logits, _, _ = self.model(x, es, eo)
        lg = logits[0].cpu()[[idx_of[m] for m in legal]]
        t = clockmodel.temperature_for(self.clock, clock, base) if self.clock else 1.0
        p = torch.softmax(lg / t, dim=0).tolist()
        out = {}
        for m, pr in zip(legal, p):
            out[mirror_move(m) if flip else m] = pr
        return out

    def choose(self, board: chess.Board, elo_self: int, elo_oppo: int, clock: float | None = None,
               base: float | None = None) -> str:
        probs = self.move_probs(board, elo_self, elo_oppo, clock, base)
        moves, ps = zip(*probs.items())
        return self.rng.choices(moves, weights=ps, k=1)[0]

    def think_time(self, clock: float | None, base: float | None) -> float:
        return clockmodel.sample_think_time(self.clock, clock, base, self.rng) if self.clock else 1.0
