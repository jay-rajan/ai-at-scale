"""The investigation graph for Temporal's LangGraph plugin (build 6b).

Same nodes and routing as langgraph_agent/graph.py (build B) and graph_in_temporal (build 6a). The difference
is where each node runs: the plugin turns every node marked execute_in="activity" into its own Temporal
activity, so the Temporal UI shows think, record, gather, decide ... as separate steps, each retried on its own.

    start (workflow) -> think -> record -> [gather] -> decide --(investigating)--> think
                                                             --(clean approve)---> auto_approve -> END
                                                             --(anything else)---> notify_reviewer -> human_review
    human_review --(approve | reject)--> END     --(investigate)--> think

Two changes from build B, both forced by the activity boundary:
- Live objects (the LLM client) can't cross it, so think and gather build their own from state["settings"].
- LangGraph retry policies aren't allowed; Temporal's RetryPolicy goes in the node metadata instead.
- Routing functions run inside the workflow, where LangGraph can't use a thread pool for sync functions,
  so each router gets a thin async wrapper.
"""
from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any

from langgraph.graph import END, START, StateGraph
from temporalio.common import RetryPolicy

import langgraph_agent.graph as lg
from agent.turn_graph import TurnContext, after_record, decide_node, gather_node, record_node, think_node

GRAPH = "invoice-investigation"


def _runtime(st: dict[str, Any]) -> Any:
    """What think_node/gather_node read: runtime.context.llm (and toolbox=None: open one per gather)."""
    return SimpleNamespace(context=TurnContext(llm=lg.OpenRouterClient(model=st["settings"].get("model"))))


async def start(st: lg.GraphState) -> lg.GraphState:
    return lg.mark_investigating(st)


async def think(st: lg.GraphState) -> lg.GraphState:
    return await think_node(st, _runtime(st))


async def gather(st: lg.GraphState) -> lg.GraphState:
    return await gather_node(st, _runtime(st))


async def route_record(st: lg.GraphState) -> str:
    return after_record(st)


async def route_decide(st: lg.GraphState) -> str:
    return lg.after_decide(st)


async def route_review(st: lg.GraphState) -> str:
    return lg.after_review(st)


def _turn(args: Any, kwargs: Any) -> str | None:
    """Label for the Temporal UI, e.g. 'turn 3'. Runs in the workflow on replay: must not raise."""
    try:
        return f"turn {args[0]['inv'].get('turn', 0)}"
    except Exception:
        return None


def _activity(timeout_s: int = 60, attempts: int = 3) -> dict[str, Any]:
    return {"execute_in": "activity", "start_to_close_timeout": timedelta(seconds=timeout_s),
            "retry_policy": RetryPolicy(maximum_attempts=attempts, initial_interval=timedelta(seconds=2)),
            "summary_fn": _turn}


def build_plugin_graph() -> Any:
    """A fresh graph every call: LangGraphPlugin rewrites the nodes it is given."""
    # pyrefly: ignore[bad-specialization]  (false positive, see agent/turn_graph.py)
    g: Any = StateGraph(lg.GraphState)
    g.add_node("start", start, metadata={"execute_in": "workflow"})
    g.add_node("think", think, metadata=_activity(timeout_s=180))          # waits on the LLM
    g.add_node("record", record_node, metadata=_activity())
    g.add_node("gather", gather, metadata=_activity(timeout_s=120))        # MCP tool calls
    g.add_node("decide", decide_node, metadata=_activity())
    g.add_node("auto_approve", lg.auto_approve_node, metadata=_activity())
    g.add_node("notify_reviewer", lg.notify_reviewer_node, metadata=_activity())
    g.add_node("human_review", lg.human_review_node, metadata=_activity())  # calls interrupt()
    g.add_edge(START, "start")
    g.add_edge("start", "think")
    g.add_edge("think", "record")
    g.add_conditional_edges("record", route_record, ["gather", "decide"])
    g.add_edge("gather", "decide")
    g.add_conditional_edges("decide", route_decide, ["think", "auto_approve", "notify_reviewer"])
    g.add_edge("auto_approve", END)
    g.add_edge("notify_reviewer", "human_review")
    g.add_conditional_edges("human_review", route_review, ["think", "human_review", END])
    return g
