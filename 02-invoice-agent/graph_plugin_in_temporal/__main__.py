"""CLI for build 6b (needs `temporal server start-dev` and `python -m graph_plugin_in_temporal.worker`).

    python -m graph_plugin_in_temporal start  INV-2026-1042
    python -m graph_plugin_in_temporal status INV-2026-1042
    python -m graph_plugin_in_temporal decide INV-2026-1042 investigate "Check previous bank-account changes"

No `history` command: every node is an activity, so the history is the Temporal UI's timeline.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

from temporalio.client import Client

from agent.env import load_env
from graph_plugin_in_temporal.workflows import TASK_QUEUE, PluginInvestigationWorkflow
from workflow.workflows import HumanDecision, InvestigationRequest

load_env()


def _wid(invoice_id: str) -> str:
    return f"plugin-invoice-{invoice_id}"


async def _run(a: argparse.Namespace) -> None:
    c = await Client.connect(os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    if a.cmd == "start":
        req = InvestigationRequest(invoice_id=a.invoice_id, max_turns=int(os.environ.get("MAX_TURNS", 12)),
                                   max_cost_usd=float(os.environ.get("MAX_COST_USD", 0.5)),
                                   strategy=os.environ.get("CONTEXT_STRATEGY", "structured"), model=os.environ.get("AGENT_MODEL"))
        h = await c.start_workflow(PluginInvestigationWorkflow.run, req, id=_wid(a.invoice_id), task_queue=TASK_QUEUE)
        print(f"Started {h.id}. Check with: python -m graph_plugin_in_temporal status {a.invoice_id}")
        return
    h = c.get_workflow_handle(_wid(a.invoice_id))
    if a.cmd == "decide":
        if a.decision == "investigate" and not a.instruction:
            raise SystemExit('investigate needs an instruction, e.g. "Check previous bank-account changes"')
        await h.signal(PluginInvestigationWorkflow.submit_decision,
                       HumanDecision(decision=a.decision, instruction=a.instruction, reviewer=os.environ.get("USER", "reviewer")))
        print(f"Sent '{a.decision}' to {h.id}.")
        return
    print(json.dumps(await h.query(PluginInvestigationWorkflow.status), indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m graph_plugin_in_temporal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("start", "status"):
        sub.add_parser(c).add_argument("invoice_id")
    d = sub.add_parser("decide")
    d.add_argument("invoice_id")
    d.add_argument("decision", choices=["approve", "reject", "investigate"])
    d.add_argument("instruction", nargs="?")
    asyncio.run(_run(ap.parse_args()))


if __name__ == "__main__":
    main()
