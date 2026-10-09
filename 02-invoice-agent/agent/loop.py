"""The explicit agent loop:  observe -> reason about what's missing -> choose a tool -> execute -> update state -> repeat.

`agent_step` runs exactly ONE turn (one model call). Temporal runs each step as an activity,
so a crash mid-investigation resumes from the last completed turn. `run_investigation`
loops steps locally (CLI and tests).
"""
from __future__ import annotations

import json
from typing import Any, Protocol

from .context import build_messages, control_tools
from .state import RECOMMENDATIONS, InvestigationState
from .trace import Trace

MAX_ENTERPRISE_CALLS_PER_TURN = 2


class ChatModel(Protocol):
    model: str
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], temperature: float = 0.0) -> Any: ...


def _snapshot(s: InvestigationState) -> dict[str, Any]:
    d = s.to_dict()
    d.pop("transcript", None)            # logged separately as `messages_sent`
    for e in d["evidence"]:              # keep snapshots readable; full results are in tool_results
        e["result"] = "<see tool_results>"
    return d


async def agent_step(s: InvestigationState, toolbox: Any, llm: ChatModel, trace: Trace | None,
                     max_turns: int = 12, max_cost_usd: float = 0.50, strategy: str = "structured") -> InvestigationState:
    if s.status != "investigating":
        return s
    before = _snapshot(s)
    messages = build_messages(s, max_turns, strategy)
    tools = toolbox.as_openai_tools() + control_tools()

    resp = await llm.chat(messages, tools)
    _add_usage(s, resp.usage)
    nudged = False
    if not resp.tool_calls:                                   # model answered in prose: one nudge, same turn
        nudged = True
        resp = await llm.chat(messages + [{"role": "assistant", "content": resp.content or ""},
                                          {"role": "user", "content": "Respond by calling a tool (an enterprise tool, record_findings, or conclude)."}], tools)
        _add_usage(s, resp.usage)                             # both calls are paid for, so both are counted

    if strategy == "full_history":
        s.transcript.append({"role": "assistant", "content": resp.content or "",
                             "tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"], "arguments": c["raw_arguments"] if isinstance(c["raw_arguments"], str) else json.dumps(c["arguments"])}}
                                            for c in resp.tool_calls]})

    tool_results: list[dict[str, Any]] = []
    calls = resp.tool_calls
    findings = [c for c in calls if c["name"] == "record_findings"]
    enterprise = [c for c in calls if c["name"] not in ("record_findings", "conclude")]
    conclusions = [c for c in calls if c["name"] == "conclude"]

    for c in findings:                                         # 1. update beliefs from what was already seen
        a = c["arguments"]
        for f in a.get("facts") or []:
            if isinstance(f, dict) and f.get("statement"):
                s.add_fact(f["statement"], f.get("evidence_ids") or [])
        for r in a.get("risk_flags") or []:
            if isinstance(r, dict) and r.get("flag"):
                s.add_risk_flag(r["flag"], r.get("severity", "medium"), r.get("evidence_ids") or [])
        for q in a.get("new_open_questions") or []:
            if isinstance(q, str) and q.strip():
                s.add_question(q.strip())
        for qid in a.get("resolved_question_ids") or []:
            s.resolve_question(qid)
        _tool_msg(s, strategy, c, {"ok": True})

    for i, c in enumerate(enterprise):                         # 2. gather new evidence
        name, args = c["name"], c["arguments"]
        if i >= MAX_ENTERPRISE_CALLS_PER_TURN:
            out = {"skipped": f"max {MAX_ENTERPRISE_CALLS_PER_TURN} enterprise calls per turn"}
            tool_results.append({"tool": name, "args": args, "skipped": True, "result": out})
            _tool_msg(s, strategy, c, out)
            continue
        if name not in toolbox.names:
            result, ok, ms = {"error": f"unknown tool {name}"}, False, 0
        elif (dup := s.already_called(name, args)):
            out = {"duplicate": True, "already_in_evidence": dup}
            tool_results.append({"tool": name, "args": args, "duplicate_of": dup, "result": out})
            _tool_msg(s, strategy, c, out)
            continue
        else:
            result, ok, ms = await toolbox.call(name, args)
        eid = s.add_evidence(name, args, result)
        s.tool_history.append({"turn": s.turn + 1, "tool": name, "args": args, "evidence_id": eid, "ok": ok})
        tool_results.append({"tool": name, "args": args, "evidence_id": eid, "ok": ok, "latency_ms": ms, "result": result})
        _tool_msg(s, strategy, c, {"evidence_id": eid, "result": result})

    s.turn += 1
    guardrail = None
    if conclusions and not enterprise:                          # 3. conclude (only if it didn't also ask for more data)
        a = conclusions[-1]["arguments"]
        rec = a.get("recommendation") if a.get("recommendation") in RECOMMENDATIONS else "human_review"
        model_rec = rec
        open_flags = [r for r in s.risk_flags if r["severity"] in ("medium", "high")]
        if rec == "approve" and open_flags:                     # code, not the model, has the last word on auto-approval
            rec, guardrail = "human_review", f"approve blocked: {len(open_flags)} medium/high risk flag(s) recorded"
        s.recommendation = {"recommendation": rec, "model_recommendation": model_rec, "guardrail": guardrail,
                            "summary": a.get("summary", ""), "key_evidence_ids": a.get("key_evidence_ids") or [],
                            "confidence": a.get("confidence"), "turn": s.turn}
        s.status, s.stop_reason = "concluded", "concluded"
        _tool_msg(s, strategy, conclusions[-1], {"ok": True})
    else:
        for c in conclusions:                                   # conclude sent together with a data request: not applied
            _tool_msg(s, strategy, c, {"not_applied": "you also requested more data; review it, then conclude"})
    if s.status == "concluded":
        pass
    elif not calls:
        s.status, s.stop_reason = "stopped", "no_tool_call"
    elif s.turn >= max_turns:
        s.status, s.stop_reason = "stopped", "max_turns"
    elif s.usage["cost_usd"] >= max_cost_usd:
        s.status, s.stop_reason = "stopped", "cost_budget"

    if trace:
        trace.write("turn", turn=s.turn, model=llm.model, strategy=strategy, nudged=nudged,
                    messages_sent=messages, tools_offered=[t["function"]["name"] for t in tools],
                    response={"content": resp.content, "finish_reason": resp.finish_reason,
                              "tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in calls],
                              "provider": resp.provider, "latency_ms": resp.latency_ms},
                    usage=resp.usage, tool_results=tool_results, guardrail=guardrail,
                    state_before=before, state_after=_snapshot(s), stop_reason=s.stop_reason)
    return s


def _add_usage(s: InvestigationState, u: dict[str, Any]) -> None:
    s.usage["llm_calls"] += 1
    for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens"):
        s.usage[k] = s.usage.get(k, 0) + int(u.get(k) or 0)
    s.usage["cost_usd"] = round(s.usage["cost_usd"] + float(u.get("cost_usd") or 0), 6)


def _tool_msg(s: InvestigationState, strategy: str, call: dict[str, Any], content: Any) -> None:
    if strategy == "full_history":
        s.transcript.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(content, default=str)})


def resume_with_instruction(s: InvestigationState, decision: str, instruction: str | None, strategy: str = "structured") -> None:
    """A human asked for more investigation: keep everything learned so far, add the instruction, continue."""
    s.add_human_message(decision, instruction)
    s.status, s.stop_reason = "investigating", None
    if strategy == "full_history":
        s.transcript.append({"role": "user", "content": f"Human reviewer instruction: {instruction}"})


async def run_investigation(s: InvestigationState, toolbox: Any, llm: ChatModel, trace: Trace | None,
                            max_turns: int = 12, max_cost_usd: float = 0.50, strategy: str = "structured") -> InvestigationState:
    if trace:
        trace.write("run_start", model=llm.model, strategy=strategy, max_turns=max_turns, state=_snapshot(s))
    while s.status == "investigating":
        s = await agent_step(s, toolbox, llm, trace, max_turns, max_cost_usd, strategy)
    if trace:
        trace.write("run_end", stop_reason=s.stop_reason, recommendation=s.recommendation, usage=s.usage,
                    turns=s.turn, final_state=_snapshot(s))
    return s
