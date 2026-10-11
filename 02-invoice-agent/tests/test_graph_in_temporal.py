"""Build C1 (the LangGraph graph inside one Temporal activity), with the fake model. Free, offline."""
from __future__ import annotations

import asyncio
import uuid

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

import agent.turn_graph as tg
import langgraph_agent.graph as lg
from graph_in_temporal.activities import run_graph
from graph_in_temporal.workflows import GraphInvestigationWorkflow
from tests.fake_llm import FakeModel
from workflow.workflows import HumanDecision, InvestigationRequest


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(lg, "OpenRouterClient", lambda model=None: FakeModel())
    monkeypatch.setenv("GRAPH_DB", str(tmp_path / "graph.db"))


async def _until(h, status: str) -> dict:
    for _ in range(200):
        v = await h.query(GraphInvestigationWorkflow.status)
        if v["business_status"] == status:
            return v
        await asyncio.sleep(0.1)
    raise AssertionError(f"never reached {status}: {v}")


async def _start(env, tq: str, invoice_id: str):
    return await env.client.start_workflow(GraphInvestigationWorkflow.run, InvestigationRequest(invoice_id=invoice_id),
                                           id=f"graph-{uuid.uuid4().hex[:6]}", task_queue=tq)


async def test_graph_pauses_for_human_then_investigates_and_rejects():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        tq = f"tq-{uuid.uuid4().hex[:6]}"
        async with Worker(env.client, task_queue=tq, workflows=[GraphInvestigationWorkflow], activities=[run_graph]):
            h = await _start(env, tq, "INV-2026-1042")
            v = await _until(h, "HUMAN_REVIEW")
            turns = v["agent"]["turn"]
            await h.signal(GraphInvestigationWorkflow.submit_decision,
                           HumanDecision("investigate", "Check previous payments to this vendor."))
            await asyncio.sleep(0.5)
            v = await _until(h, "HUMAN_REVIEW")
            assert v["agent"]["turn"] > turns and v["last_activity"]["entry"] == "decision"
            await h.signal(GraphInvestigationWorkflow.submit_decision, HumanDecision("reject"))
            r = await h.result()
    assert r.business_status == "REJECTED" and [d["decision"] for d in r.decisions] == ["investigate", "reject"]


async def test_clean_invoice_is_auto_approved_in_one_activity():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        tq = f"tq-{uuid.uuid4().hex[:6]}"
        async with Worker(env.client, task_queue=tq, workflows=[GraphInvestigationWorkflow], activities=[run_graph]):
            r = await (await _start(env, tq, "INV-2026-1001")).result()
    assert r.business_status == "APPROVED" and r.entry == "start" and r.attempt == 1
    assert r.nodes.count("think") == r.agent["turn"]            # the loop ran inside LangGraph


async def test_crash_mid_graph_is_retried_by_temporal_and_resumes_from_checkpoint(monkeypatch):
    real_gather, calls = tg.gather_node, {"n": 0}

    async def crashing_gather(st, runtime):
        calls["n"] += 1
        if calls["n"] == 3:                     # turn 3: the worker "dies" as it starts fetching data
            raise RuntimeError("worker died")   # LangGraph doesn't retry RuntimeError, so it escapes the graph
        return await real_gather(st, runtime)

    monkeypatch.setattr(tg, "gather_node", crashing_gather)    # wire_turn() looks it up when the graph is built
    async with await WorkflowEnvironment.start_time_skipping() as env:
        tq = f"tq-{uuid.uuid4().hex[:6]}"
        async with Worker(env.client, task_queue=tq, workflows=[GraphInvestigationWorkflow], activities=[run_graph]):
            h = await _start(env, tq, "INV-2026-1042")
            v = await _until(h, "HUMAN_REVIEW")                 # nobody called resume: Temporal retried the activity
    assert v["last_activity"]["attempt"] == 2 and v["last_activity"]["entry"] == "resume-after-crash"
    usage = v["agent"]["usage"]
    assert usage["llm_calls"] == v["agent"]["turn"]             # no turn ran twice: turns 1-2 came from the checkpoint
