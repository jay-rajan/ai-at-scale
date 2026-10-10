"""Build B (LangGraph): same scenarios as the Temporal tests, with the fake model. Free, offline."""
from __future__ import annotations

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

import agent.turn_graph as tg
import langgraph_agent.graph as lg
from tests.fake_llm import FakeModel


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    monkeypatch.setattr(lg, "OpenRouterClient", lambda model=None: FakeModel())


def ctx():
    return lg.make_context(lg.initial_state("x")["settings"])


async def test_pauses_for_human_then_resumes_and_rejects(tmp_path):
    cfg = {"configurable": {"thread_id": "invoice-INV-2026-1042"}}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "lg.db")) as saver:
        app = lg.build_graph().compile(checkpointer=saver)
        await app.ainvoke(lg.initial_state("INV-2026-1042"), cfg, durability="sync", context=ctx())
        snap = await app.aget_state(cfg)
        assert snap.values["business_status"] == "HUMAN_REVIEW" and snap.next == ("human_review",)
        turns_before = snap.values["inv"]["turn"]

    # a different process, later: open the same checkpoint file and send the decision
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "lg.db")) as saver:
        app = lg.build_graph().compile(checkpointer=saver)
        await app.ainvoke(Command(resume={"decision": "investigate", "instruction": "Check previous payments to this vendor."}),
                          cfg, durability="sync", context=ctx())
        v = (await app.aget_state(cfg)).values
        assert v["business_status"] == "HUMAN_REVIEW" and v["inv"]["turn"] > turns_before   # resumed with prior state
        assert any(t["tool"] == "get_payment_history" for t in v["inv"]["tool_history"])     # and followed the instruction
        await app.ainvoke(Command(resume={"decision": "reject"}), cfg, durability="sync", context=ctx())
        snap = await app.aget_state(cfg)
        assert snap.values["business_status"] == "REJECTED" and snap.next == ()


async def test_clean_invoice_is_auto_approved(tmp_path):
    cfg = {"configurable": {"thread_id": "t"}}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "lg.db")) as saver:
        app = lg.build_graph().compile(checkpointer=saver)
        out = await app.ainvoke(lg.initial_state("INV-2026-1001"), cfg, durability="sync", context=ctx())
    assert out["business_status"] == "APPROVED" and out["decisions"] == []


class ProcessDied(BaseException):
    """Not an Exception, so node retries don't catch it: like the process being killed."""


async def test_crash_mid_run_resumes_from_last_checkpoint(tmp_path, monkeypatch):
    cfg = {"configurable": {"thread_id": "t"}}
    real_gather, calls = tg.gather_evidence, {"n": 0}

    async def crashing_gather(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ProcessDied()
        return await real_gather(*a, **k)

    monkeypatch.setattr(tg, "gather_evidence", crashing_gather)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "lg.db")) as saver:
        app = lg.build_graph().compile(checkpointer=saver)
        with pytest.raises((ProcessDied, BaseExceptionGroup)):   # the MCP client may wrap it in a group
            await app.ainvoke(lg.initial_state("INV-2026-1042"), cfg, durability="sync", context=ctx())
        snap = await app.aget_state(cfg)
        assert snap.next == ("gather",) and snap.values["inv"]["turn"] == 2      # two turns done, crashed in turn 3

    # nothing restarts it on its own. Something has to call it again with the same thread id:
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "lg.db")) as saver:
        app = lg.build_graph().compile(checkpointer=saver)
        await app.ainvoke(None, cfg, durability="sync", context=ctx())
        v = (await app.aget_state(cfg)).values
    assert v["business_status"] == "HUMAN_REVIEW"
    ids = [e["id"] for e in v["inv"]["evidence"]]
    assert len(ids) == len(set(ids)) and v["inv"]["usage"]["llm_calls"] == v["inv"]["turn"]   # no turn ran twice
