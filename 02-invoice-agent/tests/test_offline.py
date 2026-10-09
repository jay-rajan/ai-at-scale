"""Offline, deterministic tests: no API key, no cost. They test the system, not the model."""
import uuid

import pytest
from temporalio.client import WorkflowExecutionStatus
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from agent.loop import run_investigation
from agent.state import InvestigationState
from agent.toolbox import MCPToolbox
from tests.fake_llm import FakeModel


async def test_mcp_server_exposes_enterprise_tools():
    async with MCPToolbox() as tb:
        assert {"get_invoice", "get_purchase_order", "get_vendor", "get_vendor_bank_accounts",
                "get_vendor_change_history", "get_approval_policy", "get_employee", "get_payment_history"} <= tb.names
        result, ok, _ = await tb.call("get_invoice", {"invoice_id": "NOPE"})
        assert ok is False and result["found"] is False


@pytest.mark.parametrize("invoice_id, expected", [
    ("INV-2026-1001", "approve"), ("INV-2026-1042", "human_review"), ("INV-2026-1077", "human_review")])
async def test_loop_reaches_expected_outcome(invoice_id, expected):
    async with MCPToolbox() as tb:
        s = await run_investigation(InvestigationState(invoice_id=invoice_id), tb, FakeModel(), None)
    assert s.status == "concluded" and s.recommendation is not None
    assert s.recommendation["recommendation"] == expected
    assert all(set(r["evidence_ids"]) <= {e["id"] for e in s.evidence} for r in s.risk_flags)


async def test_guardrail_blocks_auto_approval_with_risk_flags():
    async with MCPToolbox() as tb:
        s = await run_investigation(InvestigationState(invoice_id="INV-2026-1042"), tb,
                                    FakeModel(approve_despite_flags=True), None)
    assert s.recommendation is not None
    assert s.recommendation["model_recommendation"] == "approve"
    assert s.recommendation["recommendation"] == "human_review"
    assert "blocked" in s.recommendation["guardrail"]


async def test_workflow_pauses_for_human_then_resumes_and_rejects(monkeypatch):
    import workflow.activities as acts
    from workflow.workflows import HumanDecision, InvestigationRequest, InvoiceInvestigationWorkflow
    monkeypatch.setattr(acts, "OpenRouterClient", lambda model=None: FakeModel())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        tq = f"tq-{uuid.uuid4().hex[:6]}"
        async with Worker(env.client, task_queue=tq, workflows=[InvoiceInvestigationWorkflow],
                          activities=[acts.agent_turn, acts.record_event, acts.notify_reviewer]):
            h = await env.client.start_workflow(InvoiceInvestigationWorkflow.run, InvestigationRequest(invoice_id="INV-2026-1042"),
                                                id=f"wf-{uuid.uuid4().hex[:6]}", task_queue=tq)

            async def wait_status(want):
                for _ in range(200):
                    v = await h.query(InvoiceInvestigationWorkflow.status)
                    if v.business_status == want:
                        return v
                    await env.sleep(1)
                raise AssertionError(f"never reached {want}")

            v = await wait_status("HUMAN_REVIEW")
            turns_before = v.agent["turn"]
            await env.sleep(60 * 60 * 24 * 3)             # wait three days: the workflow just sits there
            assert (await h.query(InvoiceInvestigationWorkflow.status)).business_status == "HUMAN_REVIEW"

            await h.signal(InvoiceInvestigationWorkflow.submit_decision,
                           HumanDecision(decision="investigate", instruction="Check previous payments to this vendor."))
            for _ in range(200):
                v = await h.query(InvoiceInvestigationWorkflow.status)
                if v.business_status == "HUMAN_REVIEW" and v.agent["turn"] > turns_before:
                    break
                await env.sleep(1)
            assert v.agent["turn"] > turns_before                                      # it resumed with prior state
            assert any(t["tool"] == "get_payment_history" for t in v.agent["tool_history"])  # and followed the instruction
            assert v.agent["human_messages"][0]["instruction"].startswith("Check previous")

            await h.signal(InvoiceInvestigationWorkflow.submit_decision, HumanDecision(decision="reject"))
            result = await h.result()
            assert result["business_status"] == "REJECTED"
            assert (await h.describe()).status == WorkflowExecutionStatus.COMPLETED


async def test_clean_invoice_is_auto_approved_by_workflow(monkeypatch):
    import workflow.activities as acts
    from workflow.workflows import InvestigationRequest, InvoiceInvestigationWorkflow
    monkeypatch.setattr(acts, "OpenRouterClient", lambda model=None: FakeModel())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        tq = f"tq-{uuid.uuid4().hex[:6]}"
        async with Worker(env.client, task_queue=tq, workflows=[InvoiceInvestigationWorkflow],
                          activities=[acts.agent_turn, acts.record_event, acts.notify_reviewer]):
            result = await env.client.execute_workflow(InvoiceInvestigationWorkflow.run, InvestigationRequest(invoice_id="INV-2026-1001"),
                                                       id=f"wf-{uuid.uuid4().hex[:6]}", task_queue=tq)
    assert result["business_status"] == "APPROVED" and result["decisions"] == []
