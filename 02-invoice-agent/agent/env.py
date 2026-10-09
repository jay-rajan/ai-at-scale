"""Load .env (KEY=VALUE lines) from the project folder without extra dependencies."""
import os
from pathlib import Path


def load_env(path: Path = Path(__file__).resolve().parents[1] / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip() and not k.lstrip().startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
