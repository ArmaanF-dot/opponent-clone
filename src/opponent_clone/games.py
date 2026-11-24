"""Normalised game record shared by all sources."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Game:
    source: str            # "lichess" | "chesscom"
    game_id: str
    player: str
    color: str             # "white" | "black"
    opponent: str
    player_rating: int | None
    opponent_rating: int | None
    result: str            # "win" | "loss" | "draw"
    bucket: str            # bullet | blitz | rapid | classical
    time_control: str      # "300+3"
    played_at: int         # epoch ms
    opening: str | None = None
    eco: str | None = None
    moves: str = ""        # SAN, space separated
    clocks: list[float] = field(default_factory=list)  # seconds remaining after each ply
    pgn: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
