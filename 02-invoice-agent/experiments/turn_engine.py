"""Experiment: the old way vs the new way of running one agent turn.

    fixed  the original hand-written order in agent_step: think -> record -> gather -> decide, every turn
    graph  the turn as a LangGraph graph (agent/turn_graph.py): the model's choice picks the path,
           and gather runs only if it asked for enterprise data

The model sees exactly the same prompt either way, so the decisions should match. What can differ
is the engine's own cost: time spent outside the LLM call, and how many turns touch the tools.

    python -m experiments.turn_engine --offline --runs 20   # FakeModel: no API key, no cost, exact overhead
    python -m experiments.turn_engine                       # live: 3 invoices x 2 engines x 3 runs = 18

Results go to experiments/results/turn_engine-<timestamp>.{jsonl,md}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.env import load_env
from agent.llm import OpenRouterClient
from agent.loop import ENGINES, run_investigation
from agent.state import InvestigationState
from agent.toolbox import MCPToolbox
from agent.trace import Trace
from experiments.context_strategy import EXPECTED, OUT, _lookalike_found

load_env()


class TimedLLM:
    """Wraps any chat model and adds up the time spent waiting for it."""

    def __init__(self, inner: Any):
        self.inner, self.model, self.seconds = inner, inner.model, 0.0

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], temperature: float = 0.0) -> Any:
        t0 = time.perf_counter()
        try:
            return await self.inner.chat(messages, tools, temperature)
        finally:
            self.seconds += time.perf_counter() - t0


def _model(offline: bool, model: str | None, provider: str | None) -> Any:
    if offline:
        from tests.fake_llm import FakeModel
        return FakeModel()
    return OpenRouterClient(model=model, provider=provider)


async def one_run(invoice_id: str, engine: str, i: int, offline: bool, model: str | None, provider: str | None,
                  strategy: str) -> dict:
    run_id = f"exp-{engine}-{i}-{uuid.uuid4().hex[:6]}"
    llm = TimedLLM(_model(offline, model, provider))
    async with MCPToolbox() as tb:
        t0 = time.perf_counter()
        s = await run_investigation(InvestigationState(invoice_id=invoice_id), tb, llm,
                                    None if offline else Trace(invoice_id, run_id),
                                    max_turns=12, max_cost_usd=0.50, strategy=strategy, engine=engine)
        wall = time.perf_counter() - t0
    rec = (s.recommendation or {}).get("recommendation")
    data_turns = len({t["turn"] for t in s.tool_history})
    return {"invoice_id": invoice_id, "engine": engine, "strategy": strategy, "run": i, "run_id": run_id,
            "recommendation": rec, "correct": rec == EXPECTED[invoice_id], "stop_reason": s.stop_reason,
            "turns": s.turn, "data_turns": data_turns, "no_data_turns": s.turn - data_turns,
            "wall_s": round(wall, 4), "llm_s": round(llm.seconds, 4),
            "outside_llm_ms_per_turn": round((wall - llm.seconds) * 1000 / max(s.turn, 1), 2),
            "llm_calls": s.usage["llm_calls"], "prompt_tokens": s.usage["prompt_tokens"],
            "completion_tokens": s.usage["completion_tokens"], "cost_usd": s.usage["cost_usd"],
            "evidence": [e["id"] + ":" + e["tool"] for e in s.evidence], "risk_flags": len(s.risk_flags),
            "lookalike_found": _lookalike_found(s) if invoice_id == "INV-2026-1042" else None}


def summarize(rows: list[dict], meta: dict) -> str:
    mode = "offline (FakeModel, no cost)" if meta["offline"] else f"model `{meta['model']}`, provider `{meta['provider'] or 'unpinned'}`"
    lines = [f"# Turn engine experiment: fixed vs graph, {meta['started']}", "",
             f"{mode}, strategy `{meta['strategy']}`, {meta['runs']} runs per cell, max 12 turns. "
             f"Total cost ${sum(r['cost_usd'] for r in rows):.2f}.", "",
             "| Invoice | Engine | Correct | Turns | Turns with no data asked | Wall s | LLM s | Outside LLM, ms/turn | Cost |",
             "|---|---|---|---|---|---|---|---|---|"]
    for inv in EXPECTED:
        for en in ENGINES:
            rs = [r for r in rows if r["invoice_id"] == inv and r["engine"] == en]
            if not rs:
                continue
            m = lambda k: statistics.mean(r[k] for r in rs)
            lines.append(f"| {inv} | {en} | {sum(r['correct'] for r in rs)}/{len(rs)} | {m('turns'):.1f} | "
                         f"{m('no_data_turns'):.1f} | {m('wall_s'):.2f} | {m('llm_s'):.2f} | "
                         f"{m('outside_llm_ms_per_turn'):.1f} | ${m('cost_usd'):.3f} |")
    lines += ["", "| Engine | Correct | Mean turns | Outside LLM, ms/turn (median) | Share of wall time in LLM | Total cost |",
              "|---|---|---|---|---|---|"]
    for en in ENGINES:
        rs = [r for r in rows if r["engine"] == en]
        if rs:
            share = sum(r["llm_s"] for r in rs) / max(sum(r["wall_s"] for r in rs), 1e-9)
            lines.append(f"| {en} | {sum(r['correct'] for r in rs)}/{len(rs)} | {statistics.mean(r['turns'] for r in rs):.1f} | "
                         f"{statistics.median(r['outside_llm_ms_per_turn'] for r in rs):.1f} | {share:.1%} | "
                         f"${sum(r['cost_usd'] for r in rs):.2f} |")
    if meta["offline"]:
        same = all(len({json.dumps([r["recommendation"], r["evidence"], r["risk_flags"]])
                        for r in rows if r["invoice_id"] == inv}) == 1 for inv in EXPECTED)
        lines += ["", f"Same recommendation, evidence and flags under both engines (deterministic model): **{'yes' if same else 'NO'}**."]
    return "\n".join(lines) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use the deterministic FakeModel (no key, no cost)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL"))
    ap.add_argument("--provider", default=os.environ.get("AGENT_PROVIDER", "Google"))
    ap.add_argument("--strategy", default="structured", choices=["structured", "full_history"])
    ap.add_argument("--invoices", default=",".join(EXPECTED), help="comma-separated invoice ids")
    a = ap.parse_args()
    if a.offline:
        a.concurrency = 1                       # timing without contention
    tag = "offline" if a.offline else (a.model or "default").split("/")[-1]
    started = datetime.now().strftime("%Y-%m-%d-%H%M") + "-" + tag
    OUT.mkdir(parents=True, exist_ok=True)
    jsonl = OUT / f"turn_engine-{started}.jsonl"
    # interleave the engines so drift in API latency hits both equally
    jobs = [(inv, en, i) for i in range(1, a.runs + 1) for inv in a.invoices.split(",") for en in ENGINES]
    sem, rows = asyncio.Semaphore(a.concurrency), []

    async def go(inv: str, en: str, i: int) -> None:
        async with sem:
            try:
                r = await one_run(inv, en, i, a.offline, a.model, a.provider, a.strategy)
            except Exception as e:  # keep the experiment going; record the failure
                r = {"invoice_id": inv, "engine": en, "run": i, "error": f"{type(e).__name__}: {e}"}
            rows.append(r)
            with jsonl.open("a") as f:
                f.write(json.dumps(r) + "\n")
            if not a.offline:
                print(f"[{len(rows)}/{len(jobs)}] {inv} {en} #{i}: {r.get('recommendation')} turns={r.get('turns')} "
                      f"wall={r.get('wall_s')}s ${r.get('cost_usd')} {r.get('error', '')}", flush=True)

    if a.offline:                               # one warm-up per engine (imports, graph compile, MCP spawn)
        for en in ENGINES:
            await one_run("INV-2026-1001", en, 0, True, None, None, a.strategy)
    await asyncio.gather(*(go(*j) for j in jobs))
    ok = [r for r in rows if "error" not in r]
    md = summarize(ok, {"started": started, "offline": a.offline, "strategy": a.strategy, "runs": a.runs,
                        "model": None if a.offline else OpenRouterClient(model=a.model).model, "provider": a.provider})
    if len(ok) < len(rows):
        md += f"\n{len(rows) - len(ok)} run(s) failed; see {jsonl.name}.\n"
    (OUT / f"turn_engine-{started}.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    asyncio.run(main())
