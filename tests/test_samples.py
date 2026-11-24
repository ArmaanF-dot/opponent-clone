from opponent_clone.samples import game_samples, split_games


def g(color="white", t=0, moves="e4 e5 Nf3 Nc6"):
    return {"source": "lichess", "game_id": str(t), "color": color, "player_rating": 2000,
            "opponent_rating": 1900, "moves": moves, "clocks": [], "played_at": t}


def test_white_samples():
    s = game_samples(g())
    assert [x["move"] for x in s] == ["e2e4", "g1f3"]
    assert s[0]["fen"].startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP")


def test_black_samples_and_bad_san():
    s = game_samples(g("black", moves="e4 e5 Nf3 Zz9"))
    assert [x["move"] for x in s] == ["e7e5"]  # stops at the corrupt move


def test_split_is_chronological():
    tr, ho = split_games([g(t=i) for i in range(10)], 0.2)
    assert max(x["played_at"] for x in tr) < min(x["played_at"] for x in ho) and len(ho) == 2
