"""CLI for the LangGraph build. State lives in langgraph.db (SQLite), one thread per invoice.

    python -m langgraph_agent start   INV-2026-1042        run until it finishes or needs a human
    python -m langgraph_agent status  INV-2026-1042        business status, findings, what it's waiting for
    python -m langgraph_agent decide  INV-2026-1042 investigate "Check previous bank-account changes"
    python -m langgraph_agent decide  INV-2026-1042 reject
    python -m langgraph_agent resume  INV-2026-1042        continue after a crash, from the last checkpoint
    python -m langgraph_agent history INV-2026-1042        every checkpoint: step, node, turn, status
    python -m langgraph_agent export  INV-2026-1042        write every checkpoint as JSON (logs/langgraph-<invoice>.json)
    python -m langgraph_agent graph                        write the graph as Mermaid (langgraph_agent/graph.mmd)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from agent.env import load_env
from langgraph_agent.graph import build_graph, initial_state, make_context

load_env()
DB = Path(__file__).resolve().parents[1] / "langgraph.db"


def _cfg(invoice_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": f"invoice-{invoice_id}"}}


async def _run(a: argparse.Namespace) -> None:
    if a.cmd == "graph":
        out = Path(__file__).with_name("graph.mmd")
        out.write_text(build_graph().compile().get_graph().draw_mermaid())
        print(f"Wrote {out}")
        return
    async with AsyncSqliteSaver.from_conn_string(str(DB)) as saver:
        app = build_graph().compile(checkpointer=saver)
        cfg = _cfg(a.invoice_id)
        if a.cmd == "start":
            if (await app.aget_state(cfg)).values:
                raise SystemExit(f"invoice-{a.invoice_id} already exists; use status, decide or resume")
            st = initial_state(a.invoice_id, int(os.environ.get("MAX_TURNS", 12)), float(os.environ.get("MAX_COST_USD", 0.5)),
                               os.environ.get("CONTEXT_STRATEGY", "structured"), os.environ.get("AGENT_MODEL"))
            await app.ainvoke(st, cfg, durability="sync", context=make_context(st["settings"]))
        elif a.cmd == "decide":
            if a.decision == "investigate" and not a.instruction:
                raise SystemExit('investigate needs an instruction, e.g. "Check previous bank-account changes"')
            await app.ainvoke(Command(resume={"decision": a.decision, "instruction": a.instruction,
                                              "reviewer": os.environ.get("USER", "reviewer")}), cfg, durability="sync",
                              context=await _context(app, cfg))
        elif a.cmd == "resume":
            await app.ainvoke(None, cfg, durability="sync", context=await _context(app, cfg))
        elif a.cmd == "export":
            snaps = [x async for x in app.aget_state_history(cfg)][::-1]
            rows, prev, ran = [], {}, None
            for x in snaps:
                v = x.values or {}
                inv, resp = v.get("inv") or {}, v.get("resp") or {}
                pinv = prev.get("inv") or {}
                rows.append({
                    "step": x.metadata.get("step"), "ran": ran,
                    "next": list(x.next), "business_status": v.get("business_status"), "turn": inv.get("turn"),
                    "cost_usd": (inv.get("usage") or {}).get("cost_usd", 0), "evidence": len(inv.get("evidence") or []),
                    "new_evidence": [f"{e['id']} {e['tool']}" for e in (inv.get("evidence") or [])[len(pinv.get("evidence") or []):]],
                    "new_facts": [f["statement"] for f in (inv.get("facts") or [])[len(pinv.get("facts") or []):]],
                    "new_flags": [f"[{r['severity']}] {r['flag']}" for r in (inv.get("risk_flags") or [])[len(pinv.get("risk_flags") or []):]],
                    "tool_calls": [c["name"] for c in resp.get("tool_calls") or []] if ran == "think" else [],
                    "recommendation": (inv.get("recommendation") or {}).get("recommendation"),
                    "decisions": v.get("decisions") or [], "waiting": bool(x.interrupts)})
                prev, ran = v, (x.next[0] if x.next else None)
            out = Path(__file__).resolve().parents[1] / "logs" / f"langgraph-{a.invoice_id}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"invoice_id": a.invoice_id, "checkpoints": rows}, indent=1, default=str))
            print(f"Wrote {len(rows)} checkpoints to {out}")
            return
        elif a.cmd == "history":
            snaps = [x async for x in app.aget_state_history(cfg)]
            for x in reversed(snaps):
                v, inv = x.values or {}, (x.values or {}).get("inv") or {}
                print(f"step {x.metadata.get('step'):>3}  next={','.join(x.next) or '-':<16} "
                      f"status={v.get('business_status', '-'):<13} turn={inv.get('turn', '-')}  "
                      f"evidence={len(inv.get('evidence') or [])}  ${(inv.get('usage') or {}).get('cost_usd', 0):.4f}")
            return
        _print_status(await app.aget_state(cfg), a.invoice_id)


async def _context(app: Any, cfg: dict[str, Any]) -> Any:
    v = (await app.aget_state(cfg)).values
    if not v:
        raise SystemExit("no such investigation; start it first")
    return make_context(v["settings"])


def _print_status(snap: Any, invoice_id: str) -> None:
    v = snap.values or {}
    inv = v.get("inv") or {}
    rec = inv.get("recommendation") or {}
    print(f"\n{invoice_id}: {v.get('business_status')}   (agent: {inv.get('status')}, turn {inv.get('turn')}, "
          f"stop: {inv.get('stop_reason')}, next: {','.join(snap.next) or 'done'})")
    if rec:
        print(f"Agent recommends: {rec.get('recommendation')}" + (f"  [guardrail: {rec['guardrail']}]" if rec.get("guardrail") else ""))
        print(f"\n{rec.get('summary')}\n")
    for r in inv.get("risk_flags") or []:
        print(f"  ⚠ {r['id']} [{r['severity']}] {r['flag']}")
    if v.get("decisions"):
        print("Decisions:", [(d["decision"], d.get("instruction")) for d in v["decisions"]])
    u = inv.get("usage") or {}
    print(f"Usage: {u.get('llm_calls')} LLM calls, ${u.get('cost_usd', 0):.4f}")
    if snap.interrupts:
        print(f'\nWaiting for a human. Decide with: python -m langgraph_agent decide {invoice_id} approve | reject | investigate "<instruction>"')


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m langgraph_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("start", "status", "resume", "history", "export"):
        sub.add_parser(c).add_argument("invoice_id")
    d = sub.add_parser("decide")
    d.add_argument("invoice_id")
    d.add_argument("decision", choices=["approve", "reject", "investigate"])
    d.add_argument("instruction", nargs="?")
    sub.add_parser("graph")
    a = ap.parse_args()
    asyncio.run(_run(a))


if __name__ == "__main__":
    main()
