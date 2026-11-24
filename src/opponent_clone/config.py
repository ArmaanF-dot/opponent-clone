"""Settings come from environment variables (optionally loaded from a local .env). No secrets in code."""
import os
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def lichess_token() -> str | None:
    t = os.environ.get("LICHESS_API_TOKEN", "").strip()
    return None if not t or t == "your_lichess_token_here" else t


def user_agent() -> str:
    c = os.environ.get("CONTACT_EMAIL", "").strip()
    if not c or c == "you@example.com":
        return "opponent-clone/0.1"
    return f"opponent-clone/0.1 ({c})"
