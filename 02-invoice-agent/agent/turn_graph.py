"""One agent turn as a LangGraph graph: the single definition of a turn's flow.

    think ──(asked for data)──> record ──> gather ──> decide
          ──(no data asked)───> record ──────────────> decide

`wire_turn()` adds these nodes and edges to any graph. Two graphs use it:
- TURN (below): one turn, in memory. `agent_step()` runs it, so the local CLI and the Temporal
  activity both execute this graph.
- langgraph_agent/graph.py: the full investigation, with the loop and human review around the turn.

Live objects (LLM client, MCP connection) travel in the runtime context, never in state,
so state stays plain data that can be checkpointed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from .context import control_tools
from .llm import LLMError
from .loop import CONTROL_TOOLS, apply_findings, decide, gather_evidence, think
from .state import InvestigationState
from .toolbox import MCPToolbox


@dataclass
class TurnContext:
    llm: Any                         # anything with `model` and `async chat(messages, tools)`
    toolbox: Any | None = None       # an open MCPToolbox; None = open one for each gather


class TurnState(TypedDict, total=False):
    inv: dict[str, Any]              # InvestigationState.to_dict()
    resp: dict[str, Any] | None      # this turn's model response, plus tool_results and guardrail
    turn_budget: int
    cost_budget: float
    strategy: str


_tool_schemas: list[dict[str, Any]] | None = None


async def _tools(ctx: TurnContext) -> list[dict[str, Any]]:
    global _tool_schemas
    if ctx.toolbox is not None:
        return ctx.toolbox.as_openai_tools() + control_tools()
    if _tool_schemas is None:                       # schemas only change with the MCP server: list once
        async with MCPToolbox() as tb:
            _tool_schemas = tb.as_openai_tools()
    return _tool_schemas + control_tools()


async def think_node(st: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
    s = InvestigationState.from_dict(st["inv"])
    resp = await think(s, await _tools(runtime.context), runtime.context.llm, st["turn_budget"], st["strategy"])
    return {"inv": s.to_dict(), "resp": resp}


def record_node(st: TurnState) -> TurnState:
    s = InvestigationState.from_dict(st["inv"])
    apply_findings(s, (st.get("resp") or {}).get("tool_calls", []), st["strategy"])
    return {"inv": s.to_dict()}


async def gather_node(st: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
    s = InvestigationState.from_dict(st["inv"])
    calls = (st.get("resp") or {}).get("tool_calls", [])
    if runtime.context.toolbox is not None:
        results = await gather_evidence(s, calls, runtime.context.toolbox, st["strategy"])
    else:
        async with MCPToolbox() as tb:
            results = await gather_evidence(s, calls, tb, st["strategy"])
    return {"inv": s.to_dict(), "resp": {**(st.get("resp") or {}), "tool_results": results}}


def decide_node(st: TurnState) -> TurnState:
    s = InvestigationState.from_dict(st["inv"])
    resp = dict(st.get("resp") or {})
    resp.setdefault("tool_results", [])
    resp["guardrail"] = decide(s, resp.get("tool_calls", []), st["turn_budget"], st["cost_budget"], st["strategy"])
    return {"inv": s.to_dict(), "resp": resp}


def after_record(st: TurnState) -> str:
    """The model's own choice picks the path: gather only if it asked for enterprise data."""
    calls = (st.get("resp") or {}).get("tool_calls", [])
    return "gather" if any(c["name"] not in CONTROL_TOOLS for c in calls) else "decide"


def _retry_llm(e: Exception) -> bool:
    """Retry timeouts and rate limits; a bad key or bad request won't fix itself."""
    return not isinstance(e, LLMError) or "HTTP 4" not in str(e) or "429" in str(e)


def wire_turn(g: Any, retries: bool = True) -> None:
    """Add one turn's nodes and edges. The caller connects an entry to `think` and an exit from `decide`.
    retries=False when something outside already retries the whole turn (a Temporal activity),
    so retry layers don't multiply."""
    g.add_node("think", think_node, retry_policy=RetryPolicy(max_attempts=3, retry_on=_retry_llm) if retries else None)
    g.add_node("record", record_node)
    g.add_node("gather", gather_node, retry_policy=RetryPolicy(max_attempts=3) if retries else None)
    g.add_node("decide", decide_node)
    g.add_edge("think", "record")
    g.add_conditional_edges("record", after_record, ["gather", "decide"])
    g.add_edge("gather", "decide")


def build_turn_graph() -> Any:
    # pyrefly: ignore[bad-specialization]  (false positive: LangGraph accepts TypedDict state at runtime)
    g: Any = StateGraph(TurnState, context_schema=TurnContext)
    wire_turn(g, retries=False)        # the LLM client retries HTTP errors; Temporal retries the turn
    g.add_edge(START, "think")
    g.add_edge("decide", END)
    return g.compile()


TURN = build_turn_graph()
