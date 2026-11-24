from opponent_clone.bot import allowed, board_from_state, my_clock


def test_board_replay_and_turn():
    b = board_from_state(None, "e2e4 e7e5 g1f3")
    assert b.turn is False and b.fullmove_number == 2


def test_clock_extraction():
    st = {"wtime": 61500, "btime": 30000, "winc": 2000, "binc": 2000}
    assert my_clock(st, True) == (61.5, 2.0) and my_clock(st, False) == (30.0, 2.0)


def test_allowlist_denies_by_default():
    assert not allowed("Someone", "") and not allowed("Someone", None)
    assert allowed("Armaan", "armaan, other")
