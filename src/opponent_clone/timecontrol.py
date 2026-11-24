"""Time-control bucketing: estimated duration = base + 40 * increment (seconds)."""
from __future__ import annotations

BUCKETS = ("bullet", "blitz", "rapid", "classical")


def parse_time_control(tc: str | None) -> tuple[int, int] | None:
    """'300+3' -> (300, 3); '180' -> (180, 0). None for correspondence ('1/86400') or junk."""
    if not tc or "/" in tc or tc in ("-", "?"):
        return None
    try:
        base, _, inc = tc.partition("+")
        return int(base), int(inc or 0)
    except ValueError:
        return None


def bucket(base: int, increment: int) -> str:
    est = base + 40 * increment
    if est < 180:
        return "bullet"
    if est < 480:
        return "blitz"
    if est < 1500:
        return "rapid"
    return "classical"
