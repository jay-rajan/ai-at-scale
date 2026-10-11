"""The Temporal side of build C1 is small: run the graph, wait for a human, run it again.

    run_graph ──▶ waiting? ──no──▶ done
        ▲            │yes
        └─ decision ◀┘ (signal; may wait for days, nothing runs)

The agent loop, the turn and the business states all live in the LangGraph graph.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from graph_in_temporal.activities import GraphResult, GraphRun, run_graph
    from workflow.workflows import HumanDecision, InvestigationRequest

TASK_QUEUE = "invoice-graph"


@workflow.defn
class GraphInvestigationWorkflow:
    def __init__(self) -> None:
        self.business_status = "RECEIVED"
        self.last: GraphResult | None = None
        self.pending: HumanDecision | None = None

    @workflow.run
    async def run(self, req: InvestigationRequest) -> GraphResult:
        info = workflow.info()
        run = GraphRun(thread_id=f"{info.workflow_id}/{info.run_id}", invoice_id=req.invoice_id,
                       settings={"max_turns": req.max_turns, "max_cost_usd": req.max_cost_usd, "strategy": req.strategy,
                                 "model": req.model, "max_review_rounds": req.max_review_rounds})
        self.business_status = "INVESTIGATING"
        self.last = await self._run_graph(run)
        while self.last.waiting:
            self.business_status = "HUMAN_REVIEW"
            await workflow.wait_condition(lambda: self.pending is not None)       # may wait for days
            d, self.pending = self.pending, None
            assert d is not None
            self.business_status = "INVESTIGATING" if d.decision == "investigate" else "DECIDING"
            run.decision = {"decision": d.decision, "instruction": d.instruction, "reviewer": d.reviewer}
            run.decisions_before = len(self.last.decisions)
            self.last = await self._run_graph(run)
        self.business_status = self.last.business_status
        return self.last

    async def _run_graph(self, run: GraphRun) -> GraphResult:
        return await workflow.execute_activity(
            run_graph, run,
            start_to_close_timeout=timedelta(minutes=30),     # a whole investigation, not one turn
            heartbeat_timeout=timedelta(seconds=30),          # a dead worker is noticed within 30 s
            retry_policy=RetryPolicy(maximum_attempts=5, initial_interval=timedelta(seconds=2)))

    @workflow.signal
    def submit_decision(self, d: HumanDecision) -> None:
        if self.business_status != "HUMAN_REVIEW" or d.decision not in ("approve", "reject", "investigate"):
            workflow.logger.warning("Ignoring decision %s: workflow is %s", d.decision, self.business_status)
            return
        self.pending = d

    @workflow.query
    def status(self) -> dict[str, Any]:
        last = self.last
        return {"business_status": self.business_status, "thread_id": last.thread_id if last else None,
                "agent": last.agent if last else {}, "decisions": last.decisions if last else [],
                "last_activity": {"entry": last.entry, "attempt": last.attempt, "nodes": len(last.nodes)} if last else None}
