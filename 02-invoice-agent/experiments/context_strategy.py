"""Experiment: structured brief vs full chat history, on the same invoices, same model, same provider.

    python -m experiments.context_strategy                 # 3 invoices x 2 strategies x 3 runs
    python -m experiments.context_strategy --runs 1        # quick check

Each run is a normal local investigation (agent loop + MCP + OpenRouter), with its own trace in logs/.
Results go to experiments/results/context_strategy-<timestamp>.{jsonl,md}.
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

from agent.env import load_env
from agent.llm import OpenRouterClient
from agent.loop import run_investigation
from agent.state import InvestigationState
from agent.toolbox import MCPToolbox
from agent.trace import Trace

load_env()

EXPECTED = {"INV-2026-1001": "approve", "INV-2026-1042": "human_review", "INV-2026-1077": "human_review"}
STRATEGIES = ("structured", "full_history")
OUT = Path(__file__).resolve().parent / "results"


def _lookalike_found(s: InvestigationState) -> bool:
    """Did the agent notice the look-alike remit email domain on INV-2026-1042? (No policy covers it.)"""
    text = json.dumps([s.risk_flags, (s.recommendation or {}).get("summary", "")]).lower()
    return "industria1" in text or "look-alike" in text or "lookalike" in text or "typosquat" in text


async def one_run(invoice_id: str, strategy: str, i: int, model: str | None, provider: str | None,
                  cache: bool = False) -> dict:
    run_id = f"exp-{strategy}-{i}-{uuid.uuid4().hex[:6]}"
    t0 = time.perf_counter()
    async with MCPToolbox() as tb:
        s = await run_investigation(InvestigationState(invoice_id=invoice_id), tb,
                                    OpenRouterClient(model=model, provider=provider, prompt_cache=cache),
                                    Trace(invoice_id, run_id), max_turns=12, max_cost_usd=0.50, strategy=strategy)
    rec = (s.recommendation or {}).get("recommendation")
    return {"invoice_id": invoice_id, "strategy": strategy, "run": i, "run_id": run_id, "cache": cache,
            "recommendation": rec, "correct": rec == EXPECTED[invoice_id], "stop_reason": s.stop_reason,
            "turns": s.turn, "wall_s": round(time.perf_counter() - t0, 1),
            "llm_calls": s.usage["llm_calls"], "prompt_tokens": s.usage["prompt_tokens"],
            "cached_tokens": s.usage.get("cached_tokens", 0), "completion_tokens": s.usage["completion_tokens"],
            "reasoning_tokens": s.usage.get("reasoning_tokens", 0),
            "cost_usd": s.usage["cost_usd"], "facts": len(s.facts), "risk_flags": len(s.risk_flags),
            "tools": [t["tool"] for t in s.tool_history],
            "lookalike_found": _lookalike_found(s) if invoice_id == "INV-2026-1042" else None,
            "guardrail": (s.recommendation or {}).get("guardrail")}


def summarize(rows: list[dict], meta: dict) -> str:
    lines = [f"# Context strategy experiment, {meta['started']}", "",
             f"Model `{meta['model']}`, provider `{meta['provider'] or 'unpinned'}`, {meta['runs']} runs per cell, "
             f"temperature 0, max 12 turns. Total cost ${sum(r['cost_usd'] for r in rows):.2f}.", "",
             "| Invoice | Strategy | Correct | Turns (mean) | Output tokens (mean) | of which reasoning | Prompt tokens (mean) | Cached | Cost (mean) | Look-alike found |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for inv in EXPECTED:
        for st in STRATEGIES:
            rs = [r for r in rows if r["invoice_id"] == inv and r["strategy"] == st]
            if not rs:
                continue
            m = lambda k: statistics.mean(r[k] for r in rs)
            la = "" if inv != "INV-2026-1042" else f"{sum(bool(r['lookalike_found']) for r in rs)}/{len(rs)}"
            lines.append(f"| {inv} | {st} | {sum(r['correct'] for r in rs)}/{len(rs)} | {m('turns'):.1f} | "
                         f"{m('completion_tokens'):,.0f} | {m('reasoning_tokens'):,.0f} | {m('prompt_tokens'):,.0f} | {sum(r['cached_tokens'] for r in rs):,} | "
                         f"${m('cost_usd'):.3f} | {la} |")
    lines += ["", "| Strategy | Correct | Total cost | Mean turns | Mean output tokens | of which reasoning |", "|---|---|---|---|---|---|"]
    for st in STRATEGIES:
        rs = [r for r in rows if r["strategy"] == st]
        if rs:
            lines.append(f"| {st} | {sum(r['correct'] for r in rs)}/{len(rs)} | ${sum(r['cost_usd'] for r in rs):.2f} | "
                         f"{statistics.mean(r['turns'] for r in rs):.1f} | {statistics.mean(r['completion_tokens'] for r in rs):,.0f} | "
                         f"{statistics.mean(r['reasoning_tokens'] for r in rs):,.0f} |")
    return "\n".join(lines) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL"))
    ap.add_argument("--provider", default=os.environ.get("AGENT_PROVIDER", "Google"))
    ap.add_argument("--cache", action="store_true", help="mark a prompt-cache breakpoint each call")
    ap.add_argument("--invoices", default=",".join(EXPECTED), help="comma-separated invoice ids")
    ap.add_argument("--strategies", default=",".join(STRATEGIES), help="comma-separated: structured,full_history")
    a = ap.parse_args()
    started = datetime.now().strftime("%Y-%m-%d-%H%M") + "-" + (a.model or "default").split("/")[-1] + ("-cache" if a.cache else "")
    OUT.mkdir(parents=True, exist_ok=True)
    jsonl = OUT / f"context_strategy-{started}.jsonl"
    jobs = [(inv, st, i) for i in range(1, a.runs + 1) for inv in a.invoices.split(",") for st in a.strategies.split(",")]
    sem, rows = asyncio.Semaphore(a.concurrency), []

    async def go(inv: str, st: str, i: int) -> None:
        async with sem:
            try:
                r = await one_run(inv, st, i, a.model, a.provider, a.cache)
            except Exception as e:  # keep the experiment going; record the failure
                r = {"invoice_id": inv, "strategy": st, "run": i, "error": f"{type(e).__name__}: {e}"}
            rows.append(r)
            with jsonl.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[{len(rows)}/{len(jobs)}] {inv} {st} #{i}: {r.get('recommendation')} "
                  f"turns={r.get('turns')} ${r.get('cost_usd')} {r.get('error', '')}", flush=True)

    await asyncio.gather(*(go(*j) for j in jobs))
    ok = [r for r in rows if "error" not in r]
    md = summarize(ok, {"started": started, "model": OpenRouterClient(model=a.model).model,
                        "provider": a.provider, "runs": a.runs})
    if len(ok) < len(rows):
        md += f"\n{len(rows) - len(ok)} run(s) failed; see {jsonl.name}.\n"
    (OUT / f"context_strategy-{started}.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    asyncio.run(main())
