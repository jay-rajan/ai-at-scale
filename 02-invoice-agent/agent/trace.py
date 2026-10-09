"""Structured, turn-by-turn trace of an investigation, as JSON Lines.

One file per investigation run: logs/<invoice_id>/<run_id>.jsonl. Each line is one event:
  run_start · turn (model call, context sent, tools chosen, results, state before/after, usage) · run_end
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


class Trace:
    def __init__(self, invoice_id: str, run_id: str, log_dir: Path = LOG_DIR):
        self.path = log_dir / invoice_id / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.invoice_id, self.run_id = invoice_id, run_id

    def write(self, event: str, **data: Any) -> None:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, "invoice_id": self.invoice_id,
               "run_id": self.run_id, **data}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
