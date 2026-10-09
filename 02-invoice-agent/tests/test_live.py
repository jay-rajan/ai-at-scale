"""Live scenario tests with the real model. They check OUTCOMES and key evidence,
never an exact sequence of tool calls.   Run:  pytest -m live   (about $0.30)
Set LIVE_RUNS=3 to repeat each scenario and check consistency."""
import os

import pytest

from agent.env import load_env
from agent.llm import OpenRouterClient
from agent.loop import run_investigation
from agent.state import InvestigationState
from agent.toolbox import MCPToolbox

load_env()
RUNS = int(os.environ.get("LIVE_RUNS", 1))


def _text(s: InvestigationState) -> str:
    return " ".join([f["statement"] for f in s.facts] + [r["flag"] for r in s.risk_flags]
                    + [(s.recommendation or {}).get("summary", "")]).lower()


def _tools(s: InvestigationState) -> set:
    return {h["tool"] for h in s.tool_history}


SCENARIOS = {
    "INV-2026-1001": dict(outcome="approve", must_have_tools={"get_invoice", "get_purchase_order"}, mentions=[]),
    "INV-2026-1042": dict(outcome="human_review", must_have_tools={"get_vendor_change_history"},
                          mentions=[["secondary approval", "second approv", "unapproved"]]),
    "INV-2026-1077": dict(outcome="human_review", must_have_tools={"get_purchase_order"},
                          mentions=[["exceed", "over the po", "above the po"], ["amendment", "po_variance", "variance"]]),
}


@pytest.mark.live
@pytest.mark.parametrize("invoice_id", list(SCENARIOS))
@pytest.mark.parametrize("run", range(RUNS))
async def test_scenario_outcome(invoice_id, run):
    exp = SCENARIOS[invoice_id]
    async with MCPToolbox() as tb:
        s = await run_investigation(InvestigationState(invoice_id=invoice_id), tb, OpenRouterClient(), None)
    assert s.status == "concluded", f"stopped early: {s.stop_reason}"
    assert s.recommendation is not None
    assert s.recommendation["recommendation"] == exp["outcome"]
    assert exp["must_have_tools"] <= _tools(s)                  # it gathered the evidence the decision needs
    text = _text(s)
    for options in exp["mentions"]:
        assert any(o in text for o in options), f"expected one of {options} in findings"
    for r in s.risk_flags:                                      # every risk flag is backed by real evidence
        assert r["evidence_ids"], r
