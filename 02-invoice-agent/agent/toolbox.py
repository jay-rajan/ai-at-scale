"""Connects to enterprise-finance-mcp over stdio and exposes its tools in OpenAI/OpenRouter
function-calling format. The agent discovers tools at runtime; nothing is hard-coded here."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]


class MCPToolbox:
    def __init__(self) -> None:
        self._params = StdioServerParameters(command=sys.executable,
                                             args=["-m", "enterprise_finance_mcp.server"], cwd=str(ROOT))
        self._client: Client | None = None
        self.tools: list[Any] = []

    async def __aenter__(self) -> "MCPToolbox":
        self._client = Client(self._params)
        await self._client.__aenter__()
        self.tools = (await self._client.list_tools()).tools
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.__aexit__(*exc)

    @property
    def names(self) -> set[str]:
        return {t.name for t in self.tools}

    def as_openai_tools(self) -> list[dict[str, Any]]:
        out = []
        for t in self.tools:
            schema = dict(t.input_schema or {"type": "object", "properties": {}})
            schema.pop("title", None)
            out.append({"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                         "parameters": schema}})
        return out

    async def call(self, name: str, args: dict[str, Any]) -> tuple[Any, bool, int]:
        """Returns (result, ok, latency_ms). Tool errors come back as data, not exceptions,
        so the agent can see them and decide what to do."""
        assert self._client is not None
        t0 = time.perf_counter()
        try:
            r = await self._client.call_tool(name, args)
            ms = int((time.perf_counter() - t0) * 1000)
            if r.is_error:
                text = " ".join(getattr(c, "text", "") for c in r.content or [])
                return {"error": text or "tool error"}, False, ms
            result = r.structured_content
            if result is None:
                result = {"text": " ".join(getattr(c, "text", "") for c in r.content or [])}
            ok = not (isinstance(result, dict) and result.get("found") is False)
            return result, ok, ms
        except Exception as e:  # transport-level failure
            return {"error": f"{type(e).__name__}: {e}"}, False, int((time.perf_counter() - t0) * 1000)
