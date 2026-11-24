import random

import numpy as np

from opponent_clone import clockmodel as cm
from opponent_clone.samples import game_samples


def test_bucket_edges():
    assert cm.clock_bucket(290, 300) == 4
    assert cm.clock_bucket(30, 300) == 1
    assert cm.clock_bucket(5, 300) == 0
    assert cm.clock_bucket(None, 300) == 4


def test_temperature_recovers_noise():
    rng = np.random.default_rng(0)
    # moves drawn from a flatter distribution than the model's logits imply -> fitted T > 1
    items = []
    for _ in range(600):
        lg = rng.normal(size=8) * 3
        p = np.exp(lg / 2.0); p /= p.sum()
        items.append((lg, int(rng.choice(8, p=p))))
    assert cm.fit_temperature(items) >= 1.75
    sharp = []
    for _ in range(600):
        lg = rng.normal(size=8) * 3
        p = np.exp(lg / 0.7); p /= p.sum()
        sharp.append((lg, int(rng.choice(8, p=p))))
    assert cm.fit_temperature(sharp) <= 0.8


def game():
    return {"source": "x", "game_id": "1", "color": "white", "player_rating": 2000, "opponent_rating": 1900,
            "moves": "e4 e5 Nf3 Nc6 Bb5", "time_control": "300+2", "played_at": 1,
            # clocks after each ply: white 300->..., includes increment
            "clocks": [302.0, 302.0, 295.0, 298.0, 290.0]}


def test_spent_time():
    s = game_samples(game())
    assert [x["spent"] for x in s] == [0.0, 9.0, 7.0]  # 300+2-302, 302+2-295, 295+2-290
    assert s[0]["clock"] == 300.0


def test_fit_and_sample():
    samples = [{"clock": 300 - i, "base": 300, "spent": 3.0} for i in range(200)]
    m = cm.fit(samples, min_n=10)
    assert len(m["temperature"]) == 5
    t = cm.sample_think_time(m, 1.0, 300, random.Random(0))
    assert 0 <= t <= 0.5  # capped by remaining clock
    assert cm.sample_think_time(m, 250, 300, random.Random(0)) == 3.0
