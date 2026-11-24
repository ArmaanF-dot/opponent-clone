from opponent_clone import chesscom, lichess
from opponent_clone.timecontrol import bucket, parse_time_control


def test_buckets():
    assert bucket(60, 0) == "bullet"
    assert bucket(180, 0) == "blitz"
    assert bucket(120, 1) == "bullet"   # 160s
    assert bucket(180, 2) == "blitz"    # 260s
    assert bucket(600, 0) == "rapid"
    assert bucket(900, 10) == "rapid"   # 1300s
    assert bucket(1800, 0) == "classical"
    assert bucket(900, 15) == "classical"  # 1500s


def test_parse_tc():
    assert parse_time_control("300+3") == (300, 3)
    assert parse_time_control("180") == (180, 0)
    assert parse_time_control("1/86400") is None


def li(**kw):
    d = {"id": "abc", "variant": "standard", "speed": "blitz", "status": "mate", "winner": "white",
         "createdAt": 1, "clock": {"initial": 300, "increment": 3},
         "players": {"white": {"user": {"name": "Hero"}, "rating": 2000},
                     "black": {"user": {"name": "Foe"}, "rating": 1900}},
         "opening": {"name": "Sicilian Defense", "eco": "B20"}, "moves": "e4 c5", "clocks": [30000, 29900]}
    d.update(kw)
    return d


def test_lichess():
    g = lichess.parse_game(li(), "hero")
    assert (g.color, g.result, g.bucket, g.eco) == ("white", "win", "blitz", "B20")
    assert g.clocks == [300.0, 299.0]
    assert lichess.parse_game(li(variant="chess960"), "hero") is None
    assert lichess.parse_game(li(speed="correspondence"), "hero") is None
    assert lichess.parse_game(li(), "nobody") is None


PGN = ('[Event "Live"]\n[ECO "B20"]\n[ECOUrl "https://www.chess.com/openings/Sicilian-Defense-2.Nf3"]\n\n'
       '1. e4 {[%clk 0:05:00]} 1... c5 {[%clk 0:04:59.9]} 2. Nf3 {[%clk 0:04:58]} 1-0')


def cc(**kw):
    d = {"url": "https://www.chess.com/game/live/123", "rules": "chess", "rated": True, "time_class": "blitz",
         "time_control": "300", "end_time": 10, "pgn": PGN,
         "white": {"username": "Hero", "rating": 2000, "result": "win"},
         "black": {"username": "Foe", "rating": 1900, "result": "resigned"}}
    d.update(kw)
    return d


def test_chesscom():
    g = chesscom.parse_game(cc(), "hero")
    assert (g.color, g.result, g.bucket, g.game_id) == ("white", "win", "blitz", "123")
    assert g.moves == "e4 c5 Nf3"
    assert g.clocks == [300.0, 299.9, 298.0]
    assert g.opening == "Sicilian Defense" and g.eco == "B20"
    assert chesscom.parse_game(cc(rules="chess960"), "hero") is None
    assert chesscom.parse_game(cc(time_control="1/86400", time_class="daily"), "hero") is None
