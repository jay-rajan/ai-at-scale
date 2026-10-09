"""What the model sees on each turn. Two strategies, so they can be compared:

- "structured" (default): a fresh, compact brief rendered from InvestigationState every turn.
  No chat history. The model's memory is the state.
- "full_history": the classic approach. The full message transcript is resent every turn.
"""
from __future__ import annotations

import json
from typing import Any

from .state import InvestigationState

SYSTEM_PROMPT = """You are an accounts-payable investigation agent at an enterprise.
Your job: decide whether an incoming invoice can be paid safely, by gathering evidence from enterprise systems.

How you work:
- You start with only the invoice id. Decide what information is missing, then call ONE enterprise tool per turn to get it.
- After each result, call `record_findings` in the same turn as your next tool call, to record facts (with evidence ids), risk flags and open questions. Resolve questions when answered.
- Never assume. Every fact and risk flag must cite evidence ids (E1, E2, ...).
- Check the relevant approval policies before concluding. Do not invent policy rules.
- Do not repeat a tool call you have already made; use the evidence you have.
- When you have enough evidence, call `conclude` with a recommendation:
  - "approve" only if every condition in the applicable policies is met and there are no unresolved risk flags.
  - "human_review" if anything is unverified, suspicious, or requires an approval you cannot confirm.
  - "reject" only if the evidence clearly shows the invoice must not be paid (e.g. vendor inactive, invoice not found).
- If a human reviewer has given an instruction, it takes priority: investigate it, then conclude again.
- Be efficient: stop as soon as the evidence supports a decision."""


def control_tools() -> list[dict[str, Any]]:
    """Tools the agent uses to manage its own state. They don't touch enterprise systems."""
    return [
        {"type": "function", "function": {
            "name": "record_findings",
            "description": "Record what you learned from evidence: facts, risk flags, new open questions, and questions now resolved.",
            "parameters": {"type": "object", "properties": {
                "facts": {"type": "array", "items": {"type": "object", "properties": {
                    "statement": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["statement", "evidence_ids"]}},
                "risk_flags": {"type": "array", "items": {"type": "object", "properties": {
                    "flag": {"type": "string"}, "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["flag", "severity", "evidence_ids"]}},
                "new_open_questions": {"type": "array", "items": {"type": "string"}},
                "resolved_question_ids": {"type": "array", "items": {"type": "string"}}},
                "required": []}}},
        {"type": "function", "function": {
            "name": "conclude",
            "description": "Finish the investigation with a recommendation, a short summary and the key evidence.",
            "parameters": {"type": "object", "properties": {
                "recommendation": {"type": "string", "enum": ["approve", "human_review", "reject"]},
                "summary": {"type": "string", "description": "2-5 sentences a reviewer can act on."},
                "key_evidence_ids": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]}},
                "required": ["recommendation", "summary", "key_evidence_ids"]}}},
    ]


def render_brief(s: InvestigationState, max_turns: int) -> str:
    """The compact, structured context used by the default strategy."""
    def j(x: Any) -> str:
        return json.dumps(x, separators=(",", ":"))
    lines = [f"Invoice under investigation: {s.invoice_id}", f"Turn {s.turn + 1} of at most {max_turns}.", ""]
    if s.human_messages:
        lines.append("HUMAN REVIEWER INSTRUCTIONS (highest priority):")
        for h in s.human_messages:
            lines.append(f"- [{h['decision']}] {h['instruction'] or ''}")
        lines.append("")
    lines.append("FACTS:" if s.facts else "FACTS: none yet")
    lines += [f"- {f['id']}: {f['statement']} (evidence {','.join(f['evidence_ids'])})" for f in s.facts]
    lines.append("RISK FLAGS:" if s.risk_flags else "RISK FLAGS: none")
    lines += [f"- {r['id']} [{r['severity']}]: {r['flag']} (evidence {','.join(r['evidence_ids'])})" for r in s.risk_flags]
    open_q = [q for q in s.open_questions if q["status"] == "open"]
    lines.append("OPEN QUESTIONS:" if open_q else "OPEN QUESTIONS: none")
    lines += [f"- {q['id']}: {q['question']}" for q in open_q]
    lines.append("")
    lines.append("EVIDENCE (tool results):" if s.evidence else "EVIDENCE: none yet. Start by fetching the invoice.")
    lines += [f"- {e['id']} {e['tool']}({j(e['args'])}) -> {j(e['result'])}" for e in s.evidence]
    if s.recommendation and s.human_messages:
        lines += ["", f"Your previous conclusion: {s.recommendation['recommendation']}: {s.recommendation['summary']}"]
    lines += ["", "Decide the next step: call one enterprise tool (plus record_findings), or conclude."]
    return "\n".join(lines)


def build_messages(s: InvestigationState, max_turns: int, strategy: str) -> list[dict[str, Any]]:
    if strategy == "structured":
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_brief(s, max_turns)}]
    if strategy == "full_history":
        if not s.transcript:
            s.transcript.append({"role": "user", "content": f"Investigate invoice {s.invoice_id}."})
        return [{"role": "system", "content": SYSTEM_PROMPT}, *s.transcript]
    raise ValueError(f"unknown context strategy {strategy!r}")
