"""One activity runs the whole LangGraph investigation (langgraph_agent/graph.py, unchanged).

    run_graph:  open the checkpoint DB -> work out where this thread is -> stream the graph until it
                finishes or stops at human review -> return a summary

The loop (think -> record -> [gather] -> decide -> think ...) happens inside LangGraph. Temporal sees one
long activity. If the worker dies, Temporal retries the activity, possibly on another worker, and the
graph continues from its last checkpoint instead of starting over. That only works because:
  1. the checkpoints live outside the worker (SQLite here, one machine; Postgres for several), and
  2. the activity heartbeats, so Temporal notices a dead worker in seconds, not at start_to_close.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from temporalio import activity

from agent.env import load_env
from langgraph_agent.graph import Settings, build_graph, initial_state, make_context

load_env()
ROOT = Path(__file__).resolve().parents[1]
HEARTBEAT_EVERY_S = 5


def db_path() -> str:
    return os.environ.get("GRAPH_DB", str(ROOT / "langgraph-temporal.db"))


@dataclass
class GraphRun:
    thread_id: str                              # one LangGraph thread per workflow run
    invoice_id: str
    settings: dict[str, Any]                    # langgraph_agent.graph.Settings, as a plain dict for Temporal
    decision: dict[str, Any] | None = None      # the human's answer to send, if any
    decisions_before: int = 0                   # decisions recorded when it was sent (makes a retry safe)


@dataclass
class GraphResult:
    thread_id: str
    business_status: str
    waiting: bool                               # stopped at human review
    entry: str                                  # start | decision | resume-after-crash | already-done
    attempt: int
    nodes: list[str] = field(default_factory=list)
    agent: dict[str, Any] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)


@activity.defn
async def run_graph(r: GraphRun) -> GraphResult:
    cfg = {"configurable": {"thread_id": r.thread_id}}
    async with AsyncSqliteSaver.from_conn_string(db_path()) as saver:
        app = build_graph().compile(checkpointer=saver)
        snap = await app.aget_state(cfg)
        done_decisions = len((snap.values or {}).get("decisions") or [])
        if not snap.values:
            entry, inp = "start", initial_state(r.invoice_id, r.settings["max_turns"], r.settings["max_cost_usd"],
                                               r.settings["strategy"], r.settings["model"], r.settings["max_review_rounds"])
        elif r.decision and done_decisions == r.decisions_before and snap.interrupts:
            entry, inp = "decision", Command(resume=r.decision)
        elif snap.next:
            entry, inp = "resume-after-crash", None     # the checkpoint says where to continue
        else:
            entry, inp = "already-done", None           # a retry after the graph had finished
        activity.logger.info("%s: %s (attempt %d)", r.thread_id, entry, activity.info().attempt)

        nodes: list[str] = []
        beat = asyncio.create_task(_heartbeat(nodes))
        try:
            if entry != "already-done":
                async for update in app.astream(inp, cfg, durability="sync", context=make_context(cast(Settings, r.settings)),
                                                stream_mode="updates"):
                    nodes.extend(update)                 # node names, plus "__interrupt__" at human review
                    activity.heartbeat(nodes[-1])
        finally:
            beat.cancel()
        snap = await app.aget_state(cfg)
    v = snap.values
    inv = v["inv"]
    return GraphResult(thread_id=r.thread_id, business_status=v["business_status"], waiting=bool(snap.interrupts),
                       entry=entry, attempt=activity.info().attempt, nodes=nodes, decisions=v.get("decisions") or [],
                       agent={k: inv.get(k) for k in ("invoice_id", "status", "stop_reason", "turn", "recommendation",
                                                      "risk_flags", "usage")})


async def _heartbeat(nodes: list[str]) -> None:
    """A think node can wait on the LLM for a while; keep telling Temporal this worker is alive."""
    while True:
        await asyncio.sleep(HEARTBEAT_EVERY_S)
        activity.heartbeat(nodes[-1] if nodes else "starting")
