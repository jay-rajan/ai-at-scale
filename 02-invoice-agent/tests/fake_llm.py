"""A deterministic stand-in for the LLM, for offline tests of the plumbing (loop, MCP, Temporal, HITL).

It is NOT a fixed script: like the real agent, it looks at what's in the brief and asks for
whatever is still missing. It tests that the system works, not that the AI is smart.
"""
import json
import re
from types import SimpleNamespace


class FakeModel:
    model = "fake/rule-based"

    def __init__(self, approve_despite_flags: bool = False):
        self.approve_despite_flags = approve_despite_flags

    async def chat(self, messages, tools, temperature=0.0):
        brief = messages[-1]["content"]
        ev = {}
        for m in re.finditer(r"- (E\d+) (\w+)\((\{.*?\})\) -> (\{.*\})", brief):
            ev[m.group(2)] = (m.group(1), json.loads(m.group(4)))
        inv = ev.get("get_invoice", (None, {}))[1]
        vid = inv.get("vendor_id")

        def call(name, **args):
            return SimpleNamespace(content=None, finish_reason="tool_calls", tool_calls=[
                {"id": name, "name": name, "arguments": args, "raw_arguments": json.dumps(args)}],
                usage={"prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.0}, provider="fake", latency_ms=1)

        def conclude(rec, summary, flags=()):
            calls = []
            if flags:
                calls.append({"id": "rf", "name": "record_findings", "raw_arguments": "{}",
                              "arguments": {"risk_flags": [{"flag": f, "severity": "high", "evidence_ids": [i]} for f, i in flags]}})
            calls.append({"id": "c", "name": "conclude", "raw_arguments": "{}",
                          "arguments": {"recommendation": rec, "summary": summary, "key_evidence_ids": [i for _, i in flags]}})
            return SimpleNamespace(content=None, finish_reason="tool_calls", tool_calls=calls,
                                   usage={"prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.0}, provider="fake", latency_ms=1)

        if "get_invoice" not in ev:
            m = re.search(r"Invoice under investigation: (\S+)", brief)
            assert m, "brief is missing the invoice id"
            inv_id = m.group(1)
            return call("get_invoice", invoice_id=inv_id)
        if "get_purchase_order" not in ev:
            return call("get_purchase_order", po_id=inv["po_id"])
        if "get_vendor_change_history" not in ev:
            return call("get_vendor_change_history", vendor_id=vid)
        if "HUMAN REVIEWER INSTRUCTIONS" in brief and "get_payment_history" not in ev:
            return call("get_payment_history", vendor_id=vid)
        eid_po, po = ev["get_purchase_order"]
        eid_ch, ch = ev["get_vendor_change_history"]
        flags = []
        if inv["amount"] > po["authorized_amount"]:
            flags.append(("Invoice exceeds PO authorized amount", eid_po))
        if any(c["field"] == "bank_account" and c["secondary_approval"].get("required") and not c["secondary_approval"].get("approved_by")
               for c in ch["changes"]):
            flags.append(("Bank account change lacks secondary approval", eid_ch))
        if flags and not self.approve_despite_flags:
            return conclude("human_review", "Risks found.", flags)
        return conclude("approve", "All checks passed.", flags)
