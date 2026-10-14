"""The Temporal side of build 6b: run the graph in the workflow, stop at human review, resume on a signal.

    graph.ainvoke ──▶ interrupted? ──no──▶ done
         ▲               │yes
         └── Command(resume=decision) ◀── signal (may wait for days, nothing runs)

Unlike 6a there is no SQLite checkpointer: the graph's progress is the workflow's own event history.
After a crash, Temporal replays the workflow and every finished node's result comes from history.
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from temporalio import workflow
from temporalio.langgraph import graph

with workflow.unsafe.imports_passed_through():
    from graph_plugin_in_temporal.graph import GRAPH
    from langgraph_agent.graph import initial_state
    from workflow.workflows import HumanDecision, InvestigationRequest

TASK_QUEUE = "invoice-graph-plugin"


@workflow.defn
class PluginInvestigationWorkflow:
    def __init__(self) -> None:
        self.business_status = "RECEIVED"
        self.values: dict[str, Any] = {}
        self.pending: HumanDecision | None = None

    @workflow.run
    async def run(self, req: InvestigationRequest) -> dict[str, Any]:
        app = graph(GRAPH).compile(checkpointer=InMemorySaver())      # Temporal is the durable store
        cfg: Any = {"configurable": {"thread_id": workflow.info().workflow_id}, "recursion_limit": 500}
        self.business_status = "INVESTIGATING"
        out = await app.ainvoke(initial_state(req.invoice_id, req.max_turns, req.max_cost_usd, req.strategy,
                                              req.model, req.max_review_rounds), cfg, version="v2")
        while out.interrupts:
            self.values, self.business_status = out.value, "HUMAN_REVIEW"
            await workflow.wait_condition(lambda: self.pending is not None)        # may wait for days
            d, self.pending = self.pending, None
            assert d is not None
            self.business_status = "INVESTIGATING" if d.decision == "investigate" else "DECIDING"
            out = await app.ainvoke(Command(resume={"decision": d.decision, "instruction": d.instruction,
                                                    "reviewer": d.reviewer}), cfg, version="v2")
        self.values = out.value
        self.business_status = self.values["business_status"]
        return self.status()

    @workflow.signal
    def submit_decision(self, d: HumanDecision) -> None:
        if self.business_status != "HUMAN_REVIEW" or d.decision not in ("approve", "reject", "investigate"):
            workflow.logger.warning("Ignoring decision %s: workflow is %s", d.decision, self.business_status)
            return
        self.pending = d

    @workflow.query
    def status(self) -> dict[str, Any]:
        inv = self.values.get("inv") or {}
        return {"business_status": self.business_status, "decisions": self.values.get("decisions") or [],
                "agent": {k: inv.get(k) for k in ("invoice_id", "status", "stop_reason", "turn", "recommendation",
                                                  "risk_flags", "usage")}}
