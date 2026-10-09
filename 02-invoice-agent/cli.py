"""Invoice Investigation Agent CLI.

Local (no Temporal):
  python cli.py investigate INV-2026-1042            run the agent loop once, print the result
  python cli.py trace INV-2026-1042                  show the latest run turn by turn
  python cli.py trace INV-2026-1042 --turn 3 --full  show one turn, including the exact context sent

Durable (Temporal), see README:
  python cli.py start INV-2026-1042                  start the workflow
  python cli.py status INV-2026-1042                 current business state + findings
  python cli.py decide INV-2026-1042 approve|reject
  python cli.py decide INV-2026-1042 investigate "Check whether this vendor has had previous bank-account changes."
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from agent.env import load_env

load_env()


def _settings() -> dict:
    return {"max_turns": int(os.environ.get("MAX_TURNS", 12)),
            "max_cost_usd": float(os.environ.get("MAX_COST_USD", 0.5)),
            "strategy": os.environ.get("CONTEXT_STRATEGY", "structured")}


def print_result(s) -> None:
    rec = s.recommendation or {}
    print(f"\n=== {s.invoice_id}: {rec.get('recommendation', s.status).upper()}  (stop: {s.stop_reason}, turns: {s.turn})")
    if rec.get("guardrail"):
        print(f"    guardrail: {rec['guardrail']} (model said: {rec['model_recommendation']})")
    if rec.get("summary"):
        print(f"\n{rec['summary']}")
    if s.risk_flags:
        print("\nRisk flags:")
        for r in s.risk_flags:
            print(f"  {r['id']} [{r['severity']}] {r['flag']}  ({','.join(r['evidence_ids'])})")
    print("\nFacts:")
    for f in s.facts:
        print(f"  {f['id']} {f['statement']}  ({','.join(f['evidence_ids'])})")
    print("\nTools used: " + " → ".join(h["tool"] for h in s.tool_history))
    u = s.usage
    print(f"Usage: {u['llm_calls']} LLM calls, {u['prompt_tokens']} in / {u['completion_tokens']} out tokens, ${u['cost_usd']:.4f}")


async def cmd_investigate(a) -> None:
    from agent.llm import OpenRouterClient
    from agent.loop import resume_with_instruction, run_investigation
    from agent.state import InvestigationState
    from agent.toolbox import MCPToolbox
    from agent.trace import Trace
    cfg = _settings()
    if a.strategy:
        cfg["strategy"] = a.strategy
    run_id = f"local-{uuid.uuid4().hex[:8]}"
    trace = Trace(a.invoice_id, run_id)
    s = InvestigationState(invoice_id=a.invoice_id)
    llm = OpenRouterClient(model=a.model)
    async with MCPToolbox() as tb:
        s = await run_investigation(s, tb, llm, trace, cfg["max_turns"], cfg["max_cost_usd"], cfg["strategy"])
        print_result(s)
        if a.instruction:
            print(f"\n--- human: investigate further: {a.instruction}")
            resume_with_instruction(s, "investigate", a.instruction, cfg["strategy"])
            s = await run_investigation(s, tb, llm, trace, s.turn + cfg["max_turns"], cfg["max_cost_usd"] * 2, cfg["strategy"])
            print_result(s)
    print(f"\nTrace: {trace.path}")


def cmd_trace(a) -> None:
    from agent.trace import LOG_DIR, read
    d = LOG_DIR / a.invoice_id
    files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime) if d.exists() else []
    if not files:
        sys.exit(f"No traces for {a.invoice_id}")
    path = Path(a.file) if a.file else files[-1]
    print(f"Trace: {path}")
    for ev in read(path):
        if ev["event"] == "run_start":
            print(f"\n▶ run_start  model={ev['model']} strategy={ev['strategy']} max_turns={ev['max_turns']}")
        elif ev["event"] == "turn":
            if a.turn and ev["turn"] != a.turn:
                continue
            u = ev["usage"]
            calls = ", ".join(f"{c['name']}({json.dumps(c['arguments'])[:80]})" for c in ev["response"]["tool_calls"])
            print(f"\n— turn {ev['turn']}  {u.get('prompt_tokens')} in / {u.get('completion_tokens')} out  "
                  f"${float(u.get('cost_usd') or 0):.5f}  {ev['response']['latency_ms']} ms  stop={ev['stop_reason']}")
            print(f"  chose: {calls}")
            for r in ev["tool_results"]:
                tag = r.get("evidence_id") or ("dup of " + r["duplicate_of"] if r.get("duplicate_of") else "skipped")
                print(f"  → {r['tool']} [{tag}] {json.dumps(r['result'])[:160]}")
            b, af = ev["state_before"], ev["state_after"]
            print(f"  state: facts {len(b['facts'])}→{len(af['facts'])}, risk flags {len(b['risk_flags'])}→{len(af['risk_flags'])}, "
                  f"open questions {sum(q['status']=='open' for q in b['open_questions'])}→{sum(q['status']=='open' for q in af['open_questions'])}")
            if ev.get("guardrail"):
                print(f"  ⚠ guardrail: {ev['guardrail']}")
            if a.full:
                print("  context sent:")
                for m in ev["messages_sent"]:
                    print(f"    [{m['role']}] " + str(m.get("content"))[:4000].replace("\n", "\n      "))
        elif ev["event"] == "run_end":
            print(f"\n■ run_end  stop={ev['stop_reason']}  recommendation={(ev['recommendation'] or {}).get('recommendation')}  usage={ev['usage']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Invoice Investigation Agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("investigate", help="run the agent locally, without Temporal")
    i.add_argument("invoice_id"); i.add_argument("--model"); i.add_argument("--strategy", choices=["structured", "full_history"])
    i.add_argument("--instruction", help="simulate a human 'investigate further' instruction after the first conclusion")
    t = sub.add_parser("trace", help="show a run turn by turn")
    t.add_argument("invoice_id"); t.add_argument("--turn", type=int); t.add_argument("--full", action="store_true"); t.add_argument("--file")
    for name in ("start", "status"):
        x = sub.add_parser(name); x.add_argument("invoice_id")
    d = sub.add_parser("decide"); d.add_argument("invoice_id"); d.add_argument("decision", choices=["approve", "reject", "investigate"])
    d.add_argument("instruction", nargs="?")
    a = p.parse_args()
    if a.cmd == "investigate":
        asyncio.run(cmd_investigate(a))
    elif a.cmd == "trace":
        cmd_trace(a)
    else:
        from workflow import client_cli
        asyncio.run(getattr(client_cli, a.cmd)(a))


if __name__ == "__main__":
    main()
