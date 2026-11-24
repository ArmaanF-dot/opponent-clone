from opponent_clone.profile import build_profile, opening_tree, style_stats
from opponent_clone.report import render


def g(color="white", result="win", moves="e4 e5 Nf3 Nc6 O-O O-O", bucket="blitz", t=0, clocks=None, pr=2000, orr=2000):
    return {"source": "lichess", "game_id": str(t), "player": "H", "color": color, "opponent": "X",
            "player_rating": pr, "opponent_rating": orr, "result": result, "bucket": bucket,
            "time_control": "300+0", "played_at": t, "opening": "Open Game", "eco": "C40", "moves": moves,
            "clocks": clocks or []}


def test_opening_tree():
    games = [g(t=i) for i in range(3)] + [g(moves="d4 d5", t=9)]
    tree = opening_tree(games, "white", min_n=1)
    assert tree["e4"]["n"] == 3 and abs(tree["e4"]["share"] - 0.75) < 1e-9


def test_black_reply_tree():
    tree = opening_tree([g(color="black", t=i) for i in range(2)], "black")
    assert tree["vs 1.e4"]["e5"]["n"] == 2


def test_stats_and_split():
    games = [g(t=1, result="loss"), g(t=2, result="win"), g(t=3, bucket="bullet")]
    prof = build_profile(games)
    assert set(prof) == {"blitz", "bullet"} and prof["bullet"]["low_sample"]
    s = style_stats(games[:2])
    assert s["score"] == 0.5 and s["n_after_loss"] == 1 and s["score_after_loss"] == 1.0
    assert s["castle_short_pct"] == 1.0


def test_time_trouble():
    cl = [300 - i for i in range(10)] + [10, 9]
    s = style_stats([g(clocks=cl, moves="e4 e5 " * 6)])
    assert s["time_trouble_pct"] == 1.0


def test_render():
    out = render("Hero", build_profile([g(t=i) for i in range(3)]))
    assert "## Blitz" in out and "Prep report: Hero" in out
