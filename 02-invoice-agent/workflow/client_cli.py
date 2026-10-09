"""CLI commands that talk to the running workflow (start, status, decide)."""
import json
import os

from temporalio.client import Client
from temporalio.service import RPCError

from workflow.workflows import TASK_QUEUE, HumanDecision, InvestigationRequest, InvoiceInvestigationWorkflow


async def _client() -> Client:
    return await Client.connect(os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))


def _wid(invoice_id: str) -> str:
    return f"invoice-{invoice_id}"


async def start(a) -> None:
    c = await _client()
    req = InvestigationRequest(invoice_id=a.invoice_id, max_turns=int(os.environ.get("MAX_TURNS", 12)),
                               max_cost_usd=float(os.environ.get("MAX_COST_USD", 0.5)),
                               strategy=os.environ.get("CONTEXT_STRATEGY", "structured"),
                               model=os.environ.get("AGENT_MODEL"))
    try:
        h = await c.start_workflow(InvoiceInvestigationWorkflow.run, req, id=_wid(a.invoice_id), task_queue=TASK_QUEUE)
        print(f"Started workflow {h.id}. Check progress with: python cli.py status {a.invoice_id}")
    except RPCError as e:
        print(f"Could not start: {e}")


async def status(a) -> None:
    c = await _client()
    h = c.get_workflow_handle(_wid(a.invoice_id))
    v = await h.query(InvoiceInvestigationWorkflow.status)
    s = v.agent or {}
    rec = s.get("recommendation") or {}
    print(f"\n{a.invoice_id}: {v.business_status}   (agent: {s.get('status')}, turn {s.get('turn')}, stop: {s.get('stop_reason')})")
    if rec:
        print(f"Agent recommends: {rec.get('recommendation')}"
              + (f"  [guardrail: {rec['guardrail']}]" if rec.get("guardrail") else ""))
        print(f"\n{rec.get('summary')}\n")
    for r in s.get("risk_flags") or []:
        print(f"  ⚠ {r['id']} [{r['severity']}] {r['flag']} ({','.join(r['evidence_ids'])})")
    for f in s.get("facts") or []:
        print(f"  • {f['id']} {f['statement']} ({','.join(f['evidence_ids'])})")
    if v.decisions:
        print("\nDecisions: " + json.dumps(v.decisions))
    if v.business_status == "HUMAN_REVIEW":
        print("\nDecide with:  python cli.py decide", a.invoice_id, "approve | reject | investigate \"<instruction>\"")


async def decide(a) -> None:
    if a.decision == "investigate" and not a.instruction:
        raise SystemExit("investigate needs an instruction, e.g. \"Check previous bank-account changes\"")
    c = await _client()
    h = c.get_workflow_handle(_wid(a.invoice_id))
    await h.signal(InvoiceInvestigationWorkflow.submit_decision,
                   HumanDecision(decision=a.decision, instruction=a.instruction, reviewer=os.environ.get("USER", "reviewer")))
    print(f"Sent '{a.decision}' to {h.id}.")
