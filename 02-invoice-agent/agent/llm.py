"""Minimal OpenRouter chat client with tool calling, retries and usage accounting.

Kept deliberately small so every request and response can be logged and inspected.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]          # [{id, name, arguments(dict), raw_arguments(str)}]
    finish_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    provider: str | None = None
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 timeout_s: float = 120.0, max_retries: int = 3, provider: str | None = None,
                 prompt_cache: bool | None = None):
        self.model = model or os.environ.get("AGENT_MODEL", "google/gemini-3.5-flash")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set (put it in .env)")
        self.timeout_s, self.max_retries = timeout_s, max_retries
        # Optional provider pinning (e.g. AGENT_PROVIDER=Google). OpenRouter otherwise routes each call
        # wherever it likes, and different backends can answer differently even at temperature 0.
        self.provider = provider if provider is not None else os.environ.get("AGENT_PROVIDER") or None
        # Optional prefix caching (PROMPT_CACHE=1): mark a cache breakpoint so the provider can reuse the
        # unchanged start of the prompt on the next call. Only pays off when that prefix is long enough.
        self.prompt_cache = prompt_cache if prompt_cache is not None else os.environ.get("PROMPT_CACHE") == "1"

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                   temperature: float = 0.0) -> LLMResponse:
        if self.prompt_cache:
            messages = _mark_cache_breakpoint(messages)
        body: dict[str, Any] = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto",
                                "temperature": temperature, "usage": {"include": True}}
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                   "X-Title": "invoice-investigation-agent"}
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout_s) as http:
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    r = await http.post(OPENROUTER_URL, headers=headers, json=body)
                    if r.status_code in RETRYABLE and attempt < self.max_retries:
                        await _sleep_backoff(attempt, r.headers.get("retry-after"))
                        continue
                    if r.status_code >= 400:
                        raise LLMError(f"OpenRouter HTTP {r.status_code}: {r.text[:300]}")
                    data = r.json()
                    if "error" in data:
                        raise LLMError(f"OpenRouter error: {json.dumps(data['error'])[:300]}")
                    return _parse(data, int((time.perf_counter() - t0) * 1000))
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    last_err = e
                    if attempt < self.max_retries:
                        await _sleep_backoff(attempt, None)
                        continue
        raise LLMError(f"OpenRouter request failed after retries: {last_err}")


def _mark_cache_breakpoint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put one cache breakpoint on the last assistant message (or the system prompt on turn 1).
    Everything before it is reused next turn. Not on tool messages: via OpenRouter that inflated
    the counted prompt in our tests."""
    out = list(messages)
    idx = next((i for i in range(len(out) - 1, -1, -1) if out[i]["role"] == "assistant"),
               next((i for i, m in enumerate(out) if m["role"] == "system"), None))
    if idx is not None and isinstance(out[idx].get("content"), str | None):
        m = dict(out[idx])
        m["content"] = [{"type": "text", "text": m.get("content") or " ", "cache_control": {"type": "ephemeral"}}]
        out[idx] = m
    return out


async def _sleep_backoff(attempt: int, retry_after: str | None) -> None:
    import asyncio
    delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 20)
    await asyncio.sleep(delay)


def _parse(data: dict[str, Any], latency_ms: int) -> LLMResponse:
    choice = data["choices"][0]
    msg = choice.get("message", {})
    calls = []
    for tc in msg.get("tool_calls") or []:
        raw = tc.get("function", {}).get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {"_unparseable_arguments": raw}
        calls.append({"id": tc.get("id"), "name": tc.get("function", {}).get("name"),
                      "arguments": args if isinstance(args, dict) else {"_value": args}, "raw_arguments": raw})
    u = data.get("usage") or {}
    return LLMResponse(content=msg.get("content"), tool_calls=calls, finish_reason=choice.get("finish_reason"),
                       usage={"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                              "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
                              "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
                              "cost_usd": u.get("cost", 0.0) or 0.0},
                       model=data.get("model", ""), provider=data.get("provider"), latency_ms=latency_ms, raw=data)
