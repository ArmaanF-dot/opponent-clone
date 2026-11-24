"""Clock awareness for the clone, fitted from a player's own games.

Two small models, both per 'clock bucket' (own remaining time as a fraction of the starting time):
  * temperature: how much the player's moves spread out (T > 1 = more random / worse) when short of time.
  * think time: the empirical distribution of seconds spent per move.
Pure Python + numpy; no torch needed to use a saved model.
"""
from __future__ import annotations

import json
import random
from bisect import bisect_right

import numpy as np

# upper edges of remaining-clock fraction; the last bucket is "everything above"
EDGES = (0.05, 0.15, 0.35, 0.65)
TEMP_GRID = (0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5)
MAX_TIMES = 400


def clock_bucket(clock: float | None, base: float | None) -> int:
    if clock is None or not base:
        return len(EDGES)  # unknown: treat as plenty of time
    return bisect_right(EDGES, clock / base)


def fit_temperature(items: list[tuple[np.ndarray, int]], grid=TEMP_GRID) -> float:
    """items = [(legal-move logits, index of the move actually played)]. Returns the T that maximises likelihood."""
    if not items:
        return 1.0
    best_t, best_ll = 1.0, -np.inf
    for t in grid:
        ll = 0.0
        for lg, k in items:
            z = lg / t
            m = z.max()
            ll += z[k] - (m + np.log(np.exp(z - m).sum()))
        if ll > best_ll:
            best_t, best_ll = t, ll
    return float(best_t)


def fit(samples: list[dict], logit_items: dict[int, list[tuple[np.ndarray, int]]] | None = None, min_n: int = 150) -> dict:
    """samples: output of samples.game_samples. logit_items: per-bucket (logits, played idx) from the model.

    Buckets with fewer than min_n samples borrow from the pooled data (shrinks toward the overall player)."""
    nb = len(EDGES) + 1
    times: list[list[float]] = [[] for _ in range(nb)]
    for s in samples:
        if s.get("spent") is not None:
            times[clock_bucket(s.get("clock"), s.get("base"))].append(s["spent"])
    pooled = [x for t in times for x in t]
    model = {"edges": list(EDGES), "think_times": [], "temperature": []}
    for b in range(nb):
        ts = times[b] if len(times[b]) >= min_n else pooled
        ts = sorted(ts)
        if len(ts) > MAX_TIMES:
            ts = [ts[int(i * (len(ts) - 1) / (MAX_TIMES - 1))] for i in range(MAX_TIMES)]
        model["think_times"].append([round(x, 2) for x in ts])
    pooled_items = [it for v in (logit_items or {}).values() for it in v]
    for b in range(nb):
        items = (logit_items or {}).get(b, [])
        model["temperature"].append(fit_temperature(items if len(items) >= min_n else pooled_items))
    model["n_per_bucket"] = [len(t) for t in times]
    return model


def temperature_for(model: dict, clock: float | None, base: float | None) -> float:
    return model["temperature"][clock_bucket(clock, base)]


def sample_think_time(model: dict, clock: float | None, base: float | None, rng: random.Random | None = None,
                      reserve: float = 0.5) -> float:
    """Seconds to wait before moving: a draw from this player's own history in this clock bucket,
    never more than the time left minus a small reserve."""
    rng = rng or random
    ts = model["think_times"][clock_bucket(clock, base)]
    t = rng.choice(ts) if ts else 1.0
    if clock is not None:
        t = min(t, max(0.0, clock - reserve))
    return t


def save(model: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(model, f)


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
