"""Explicit investigation state.

The agent never relies on the raw chat transcript to remember what it learned. Every turn,
the model gets a fresh context rendered from this state (see agent/context.py). The raw
transcript is still kept, so context strategies can be compared later.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

# Business statuses are owned by the workflow (Temporal). The agent only reports these:
AGENT_STATUSES = ("investigating", "concluded", "stopped")
RECOMMENDATIONS = ("approve", "human_review", "reject")


@dataclass
class InvestigationState:
    invoice_id: str
    invoice: dict[str, Any] | None = None                 # filled when get_invoice is called
    facts: list[dict[str, Any]] = field(default_factory=list)          # {id, statement, evidence_ids}
    evidence: list[dict[str, Any]] = field(default_factory=list)       # {id, tool, args, result, turn}
    open_questions: list[dict[str, Any]] = field(default_factory=list) # {id, question, status}
    risk_flags: list[dict[str, Any]] = field(default_factory=list)     # {id, flag, severity, evidence_ids}
    tool_history: list[dict[str, Any]] = field(default_factory=list)   # {turn, tool, args, evidence_id, ok}
    human_messages: list[dict[str, Any]] = field(default_factory=list) # {at_turn, decision, instruction}
    status: str = "investigating"
    turn: int = 0
    recommendation: dict[str, Any] | None = None          # set by the `conclude` tool
    stop_reason: str | None = None
    usage: dict[str, float] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0, "llm_calls": 0})
    transcript: list[dict[str, Any]] = field(default_factory=list)     # raw messages, for the full-history strategy only

    # ---- serialisation (Temporal passes plain dicts) ----
    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InvestigationState":
        return cls(**copy.deepcopy(d))

    # ---- mutations used by the loop ----
    def add_evidence(self, tool: str, args: dict[str, Any], result: Any) -> str:
        eid = f"E{len(self.evidence) + 1}"
        self.evidence.append({"id": eid, "tool": tool, "args": args, "result": result, "turn": self.turn})
        if tool == "get_invoice" and isinstance(result, dict) and result.get("found"):
            self.invoice = result
        return eid

    def add_fact(self, statement: str, evidence_ids: list[str]) -> None:
        if any(f["statement"] == statement for f in self.facts):
            return
        self.facts.append({"id": f"F{len(self.facts) + 1}", "statement": statement,
                           "evidence_ids": self._known(evidence_ids)})

    def add_risk_flag(self, flag: str, severity: str, evidence_ids: list[str]) -> None:
        if any(r["flag"] == flag for r in self.risk_flags):
            return
        self.risk_flags.append({"id": f"R{len(self.risk_flags) + 1}", "flag": flag,
                                "severity": severity if severity in ("low", "medium", "high") else "medium",
                                "evidence_ids": self._known(evidence_ids)})

    def add_question(self, question: str) -> None:
        if any(q["question"] == question for q in self.open_questions):
            return
        self.open_questions.append({"id": f"Q{len(self.open_questions) + 1}", "question": question, "status": "open"})

    def resolve_question(self, qid: str) -> None:
        for q in self.open_questions:
            if q["id"] == qid:
                q["status"] = "resolved"

    def add_human_message(self, decision: str, instruction: str | None) -> None:
        self.human_messages.append({"at_turn": self.turn, "decision": decision, "instruction": instruction})

    def _known(self, ids: list[str]) -> list[str]:
        valid = {e["id"] for e in self.evidence}
        return [i for i in (ids or []) if i in valid]

    def already_called(self, tool: str, args: dict[str, Any]) -> str | None:
        for h in self.tool_history:
            if h["tool"] == tool and h["args"] == args and h.get("ok"):
                return h["evidence_id"]
        return None
