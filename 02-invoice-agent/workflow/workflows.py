"""The durable business process. Temporal owns the business state; the agent owns the reasoning.

    RECEIVED → INVESTIGATING → (APPROVED) | HUMAN_REVIEW → APPROVED | REJECTED | INVESTIGATING → ...

The workflow can wait in HUMAN_REVIEW for days. No process, thread or LLM call is held
open while it waits; Temporal persists the state and wakes the workflow when a decision arrives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from agent.loop import resume_with_instruction
    from agent.state import InvestigationState
    from workflow.activities import TurnInput, WorkflowEvent, agent_turn, notify_reviewer, record_event

TASK_QUEUE = "invoice-investigations"


@dataclass
class InvestigationRequest:
    invoice_id: str
    max_turns: int = 12
    max_cost_usd: float = 0.50
    strategy: str = "structured"
    model: str | None = None
    max_review_rounds: int = 3


@dataclass
class HumanDecision:
    decision: str                      # approve | reject | investigate
    instruction: str | None = None
    reviewer: str = "reviewer"


@dataclass
class StatusView:
    business_status: str
    agent: dict[str, Any] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)


@workflow.defn
class InvoiceInvestigationWorkflow:
    def __init__(self) -> None:
        self.business_status = "RECEIVED"
        self.inv: dict[str, Any] = {}
        self.pending: HumanDecision | None = None
        self.decisions: list[dict[str, Any]] = []
        self.run_id = ""

    @workflow.run
    async def run(self, req: InvestigationRequest) -> dict[str, Any]:
        self.run_id = workflow.info().workflow_id
        self.inv = InvestigationState(invoice_id=req.invoice_id).to_dict()
        await self._event("workflow_start", {"request": req.__dict__})
        turn_budget = req.max_turns
        rounds = 0

        while True:
            await self._set_status("INVESTIGATING")
            await self._investigate(req, turn_budget)

            rec = (self.inv.get("recommendation") or {}).get("recommendation")
            if self.inv["status"] == "concluded" and rec == "approve" and not self.decisions:
                await self._set_status("APPROVED", {"by": "agent", "auto": True})
                return self._result()

            # Anything other than a clean approval goes to a human: risk, reject, max turns, budget.
            await self._set_status("HUMAN_REVIEW")
            await workflow.execute_activity(notify_reviewer, self._review_summary(),
                                            start_to_close_timeout=timedelta(seconds=30))
            await workflow.wait_condition(lambda: self.pending is not None)       # may wait for days
            d, self.pending = self.pending, None
            assert d is not None
            self.decisions.append({"decision": d.decision, "instruction": d.instruction, "reviewer": d.reviewer,
                                   "at": workflow.now().isoformat()})
            await self._event("human_decision", self.decisions[-1])

            if d.decision == "approve":
                await self._set_status("APPROVED", {"by": d.reviewer})
                return self._result()
            if d.decision == "reject":
                await self._set_status("REJECTED", {"by": d.reviewer})
                return self._result()
            rounds += 1
            if rounds > req.max_review_rounds:
                await self._set_status("REJECTED", {"by": "system", "reason": "too many investigation rounds"})
                return self._result()
            s = InvestigationState.from_dict(self.inv)
            resume_with_instruction(s, "investigate", d.instruction, req.strategy)
            self.inv = s.to_dict()
            turn_budget = s.turn + req.max_turns                      # a fresh turn budget for each round

    async def _investigate(self, req: InvestigationRequest, turn_budget: int) -> None:
        while self.inv["status"] == "investigating":
            self.inv = await workflow.execute_activity(
                agent_turn,
                TurnInput(state=self.inv, run_id=self.run_id, max_turns=turn_budget,
                          max_cost_usd=req.max_cost_usd * (len(self.decisions) + 1), strategy=req.strategy, model=req.model),
                start_to_close_timeout=timedelta(minutes=3),
                retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=2),
                                         non_retryable_error_types=["ValueError"]))

    @workflow.signal
    def submit_decision(self, d: HumanDecision) -> None:
        if self.business_status != "HUMAN_REVIEW":
            workflow.logger.warning("Ignoring decision %s: workflow is %s", d.decision, self.business_status)
            return
        if d.decision not in ("approve", "reject", "investigate"):
            workflow.logger.warning("Ignoring unknown decision %s", d.decision)
            return
        self.pending = d

    @workflow.query
    def status(self) -> StatusView:
        s = self.inv or {}
        return StatusView(business_status=self.business_status, decisions=self.decisions, agent={
            k: s.get(k) for k in ("invoice_id", "status", "stop_reason", "turn", "recommendation", "facts",
                                  "risk_flags", "open_questions", "tool_history", "human_messages", "usage")})

    async def _set_status(self, new: str, extra: dict[str, Any] | None = None) -> None:
        old, self.business_status = self.business_status, new
        await self._event("status_change", {"from": old, "to": new, **(extra or {})})

    async def _event(self, name: str, data: dict[str, Any]) -> None:
        await workflow.execute_activity(record_event, WorkflowEvent(self.inv.get("invoice_id", "unknown"), self.run_id, name, data),
                                        start_to_close_timeout=timedelta(seconds=10))

    def _review_summary(self) -> dict[str, Any]:
        r = self.inv.get("recommendation") or {}
        return {"invoice_id": self.inv["invoice_id"], "workflow_id": self.run_id, "agent_recommendation": r,
                "stop_reason": self.inv.get("stop_reason"), "risk_flags": self.inv.get("risk_flags"),
                "at": workflow.now().isoformat()}

    def _result(self) -> dict[str, Any]:
        return {"invoice_id": self.inv["invoice_id"], "business_status": self.business_status,
                "agent_recommendation": self.inv.get("recommendation"), "decisions": self.decisions,
                "usage": self.inv.get("usage"), "turns": self.inv.get("turn")}
