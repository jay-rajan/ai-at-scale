"""The full investigation as a LangGraph graph (build B): no Temporal, everything local.

    think -> record -> [gather] -> decide --(investigating)--> think
                                       --(clean approve)---> auto_approve -> END
                                       --(anything else)---> notify_reviewer -> human_review
    human_review --(approve | reject)--> END
                 --(investigate)-------> think        (same state, new instruction, fresh turn budget)

The turn (think, record, gather, decide) is wired by agent/turn_graph.wire_turn(), the same definition the
hand-written loop and the Temporal activity run, so every build follows one flow and one set of rules.
State is saved to a SQLite checkpoint after every node. `human_review` calls interrupt(): the graph stops,
nothing runs while it waits, and `Command(resume=...)` continues it later, from any process.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.llm import OpenRouterClient
from agent.loop import resume_with_instruction
from agent.state import InvestigationState
from agent.turn_graph import TurnContext, TurnState, wire_turn

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ("approve", "reject", "investigate")


class Settings(TypedDict):
    max_turns: int
    max_cost_usd: float
    strategy: str
    model: str | None
    max_review_rounds: int


class GraphState(TurnState, total=False):
    business_status: str                # RECEIVED | INVESTIGATING | HUMAN_REVIEW | APPROVED | REJECTED
    decisions: list[dict[str, Any]]
    settings: Settings


def make_context(settings: Settings) -> TurnContext:
    """Live objects for the nodes. Not checkpointed: pass it again on every invoke, including resume."""
    return TurnContext(llm=OpenRouterClient(model=settings.get("model")))


# ---- nodes ----

def auto_approve_node(st: GraphState) -> GraphState:
    return {"business_status": "APPROVED"}


def notify_reviewer_node(st: GraphState) -> GraphState:
    """A side effect, so it gets its own node: human_review re-runs from the top when resumed,
    and anything before interrupt() would run twice."""
    inv = st["inv"]
    q = ROOT / "logs" / "review_queue.jsonl"
    q.parent.mkdir(parents=True, exist_ok=True)
    with q.open("a") as f:
        f.write(json.dumps({"invoice_id": inv["invoice_id"], "engine": "langgraph",
                            "agent_recommendation": inv.get("recommendation"), "stop_reason": inv.get("stop_reason"),
                            "risk_flags": inv.get("risk_flags"), "at": datetime.now(timezone.utc).isoformat()}) + "\n")
    return {"business_status": "HUMAN_REVIEW"}


def human_review_node(st: GraphState) -> GraphState:
    inv = st["inv"]
    d = interrupt({"invoice_id": inv["invoice_id"], "agent_recommendation": inv.get("recommendation"),
                   "stop_reason": inv.get("stop_reason"), "risk_flags": inv.get("risk_flags"),
                   "ask": "approve | reject | investigate (with an instruction)"})
    decision = (d or {}).get("decision")
    if decision not in DECISIONS:
        return {}                                   # invalid answer: route back here and ask again
    decisions = list(st.get("decisions") or []) + [{
        "decision": decision, "instruction": d.get("instruction"), "reviewer": d.get("reviewer", "reviewer"),
        "at": datetime.now(timezone.utc).isoformat()}]
    if decision == "approve":
        return {"decisions": decisions, "business_status": "APPROVED"}
    if decision == "reject":
        return {"decisions": decisions, "business_status": "REJECTED"}
    rounds = sum(x["decision"] == "investigate" for x in decisions)
    if rounds > st["settings"]["max_review_rounds"]:
        return {"decisions": decisions, "business_status": "REJECTED"}
    s = InvestigationState.from_dict(inv)
    resume_with_instruction(s, "investigate", d.get("instruction"), st["settings"]["strategy"])
    return {"decisions": decisions, "inv": s.to_dict(), "business_status": "INVESTIGATING",
            "turn_budget": s.turn + st["settings"]["max_turns"],                     # fresh turn budget per round
            "cost_budget": st["settings"]["max_cost_usd"] * (len(decisions) + 1)}


# ---- routing ----

def after_decide(st: GraphState) -> str:
    inv = st["inv"]
    if inv["status"] == "investigating":
        return "think"
    rec = (inv.get("recommendation") or {}).get("recommendation")
    if inv["status"] == "concluded" and rec == "approve" and not st.get("decisions"):
        return "auto_approve"
    return "notify_reviewer"                         # anything but a clean approval goes to a human


def after_review(st: GraphState) -> str:
    status = st.get("business_status")
    if status in ("APPROVED", "REJECTED"):
        return END
    if status == "INVESTIGATING":
        return "think"
    return "human_review"


def mark_investigating(st: GraphState) -> GraphState:
    return {"business_status": "INVESTIGATING"}


def build_graph() -> Any:
    # pyrefly: ignore[bad-specialization]  (false positive, see agent/turn_graph.py)
    g: Any = StateGraph(GraphState, context_schema=TurnContext)
    g.add_node("start", mark_investigating)
    wire_turn(g)                                   # think -> record -> [gather] -> decide, with node retries
    g.add_node("auto_approve", auto_approve_node)
    g.add_node("notify_reviewer", notify_reviewer_node)
    g.add_node("human_review", human_review_node)
    g.add_edge(START, "start")
    g.add_edge("start", "think")
    g.add_conditional_edges("decide", after_decide, ["think", "auto_approve", "notify_reviewer"])
    g.add_edge("auto_approve", END)
    g.add_edge("notify_reviewer", "human_review")
    g.add_conditional_edges("human_review", after_review, ["think", "human_review", END])
    return g


def initial_state(invoice_id: str, max_turns: int = 12, max_cost_usd: float = 0.50, strategy: str = "structured",
                  model: str | None = None, max_review_rounds: int = 3) -> GraphState:
    return {"inv": InvestigationState(invoice_id=invoice_id).to_dict(), "resp": None, "business_status": "RECEIVED",
            "decisions": [], "turn_budget": max_turns, "cost_budget": max_cost_usd, "strategy": strategy,
            "settings": {"max_turns": max_turns, "max_cost_usd": max_cost_usd, "strategy": strategy,
                         "model": model, "max_review_rounds": max_review_rounds}}
