"""Build 6b (Temporal's LangGraph plugin, every node an activity), with the fake model. Free, offline."""
from __future__ import annotations

import asyncio
import uuid

import pytest
from temporalio.langgraph import LangGraphPlugin
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

import graph_plugin_in_temporal.graph as pg
import langgraph_agent.graph as lg
from graph_plugin_in_temporal.workflows import PluginInvestigationWorkflow
from tests.fake_llm import FakeModel
from workflow.workflows import HumanDecision, InvestigationRequest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(lg, "OpenRouterClient", lambda model=None: FakeModel())


def _worker(env, tq: str) -> Worker:
    return Worker(env.client, task_queue=tq, workflows=[PluginInvestigationWorkflow],
                  plugins=[LangGraphPlugin(graphs={pg.GRAPH: pg.build_plugin_graph()})])


async def _until(h, status: str) -> dict:
    for _ in range(300):
        v = await h.query(PluginInvestigationWorkflow.status)
        if v["business_status"] == status:
            return v
        await asyncio.sleep(0.1)
    raise AssertionError(f"never reached {status}: {v}")


async def _start(env, tq: str, invoice_id: str):
    return await env.client.start_workflow(PluginInvestigationWorkflow.run, InvestigationRequest(invoice_id=invoice_id),
                                           id=f"plugin-{uuid.uuid4().hex[:6]}", task_queue=tq)


async def _activities(h) -> list[str]:
    return [e.activity_task_scheduled_event_attributes.activity_type.name async for e in h.fetch_history_events()
            if e.HasField("activity_task_scheduled_event_attributes")]


async def test_every_node_is_its_own_activity_then_human_rejects():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        tq = f"tq-{uuid.uuid4().hex[:6]}"
        async with _worker(env, tq):
            h = await _start(env, tq, "INV-2026-1042")
            v = await _until(h, "HUMAN_REVIEW")
            turns = v["agent"]["turn"]
            await h.signal(PluginInvestigationWorkflow.submit_decision,
                           HumanDecision("investigate", "Check previous payments to this vendor."))
            await asyncio.sleep(0.5)
            v = await _until(h, "HUMAN_REVIEW")
            assert v["agent"]["turn"] > turns
            await h.signal(PluginInvestigationWorkflow.submit_decision, HumanDecision("reject"))
            r = await h.result()
            acts = await _activities(h)
    assert r["business_status"] == "REJECTED" and [d["decision"] for d in r["decisions"]] == ["investigate", "reject"]
    names = {a.split(".")[-1] for a in acts}
    assert {"think", "record", "gather", "decide", "notify_reviewer", "human_review"} <= names
    assert sum(a.endswith(".think") for a in acts) == r["agent"]["turn"]      # one think activity per turn


async def test_clean_invoice_is_auto_approved():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        tq = f"tq-{uuid.uuid4().hex[:6]}"
        async with _worker(env, tq):
            h = await _start(env, tq, "INV-2026-1001")
            r = await h.result()
            acts = await _activities(h)
    assert r["business_status"] == "APPROVED" and acts[-1].endswith(".auto_approve")


async def test_failed_node_is_retried_alone(monkeypatch):
    real_gather, calls = pg.gather_node, {"n": 0}

    async def flaky_gather(st, runtime):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("tool server died")
        return await real_gather(st, runtime)

    monkeypatch.setattr(pg, "gather_node", flaky_gather)
    async with await WorkflowEnvironment.start_time_skipping() as env:
        tq = f"tq-{uuid.uuid4().hex[:6]}"
        async with _worker(env, tq):
            h = await _start(env, tq, "INV-2026-1042")
            v = await _until(h, "HUMAN_REVIEW")
    assert calls["n"] > 3                                                     # the failed gather ran again
    assert v["agent"]["usage"]["llm_calls"] == v["agent"]["turn"]             # but no think (LLM call) ran twice
