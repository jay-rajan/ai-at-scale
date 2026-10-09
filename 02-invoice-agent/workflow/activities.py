"""Temporal activities: the only place where the agent touches the outside world (LLM + MCP + logs).

Each `agent_turn` call runs ONE agent step. If the worker crashes, Temporal retries that turn
and every completed turn is preserved in workflow history. The LLM never runs while the
workflow is waiting for a human.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

from agent.env import load_env
from agent.llm import LLMError, OpenRouterClient
from agent.loop import agent_step
from agent.state import InvestigationState
from agent.toolbox import MCPToolbox
from agent.trace import Trace

load_env()

# Design choice: open the MCP connection inside each turn, not once per worker.
# The MCP client's async context must be entered and exited in the same task; a turn is the natural
# unit for that. It also isolates failures: a crashed tool server affects one turn, which Temporal
# retries. Cost: roughly 0.3 s to start the stdio server per turn. Pool it only if that matters.


@dataclass
class TurnInput:
    state: dict[str, Any]
    run_id: str
    max_turns: int
    max_cost_usd: float
    strategy: str
    model: str | None = None


@activity.defn
async def agent_turn(inp: TurnInput) -> dict[str, Any]:
    s = InvestigationState.from_dict(inp.state)
    trace = Trace(s.invoice_id, inp.run_id)
    try:
        async with MCPToolbox() as tb:
            llm = OpenRouterClient(model=inp.model)
            s = await agent_step(s, tb, llm, trace, inp.max_turns, inp.max_cost_usd, inp.strategy)
        return s.to_dict()
    except LLMError as e:
        # Bad key / bad request won't fix itself on retry; rate limits and timeouts were already retried inside the client.
        raise ApplicationError(str(e), type="LLMError", non_retryable="HTTP 4" in str(e) and "429" not in str(e))


@dataclass
class WorkflowEvent:
    invoice_id: str
    run_id: str
    event: str
    data: dict[str, Any]


@activity.defn
async def record_event(ev: WorkflowEvent) -> None:
    """Business events (status changes, human decisions) go into the same trace file as the agent turns."""
    Trace(ev.invoice_id, ev.run_id).write(ev.event, **ev.data)


@activity.defn
async def notify_reviewer(summary: dict[str, Any]) -> None:
    """Stand-in for Slack/email/ticket creation. Here: append to a local review queue."""
    import json
    from pathlib import Path
    q = Path(__file__).resolve().parents[1] / "logs" / "review_queue.jsonl"
    q.parent.mkdir(parents=True, exist_ok=True)
    with q.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    activity.logger.info("Review requested for %s", summary.get("invoice_id"))
