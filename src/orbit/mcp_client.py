from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPError(RuntimeError):
    pass


@dataclass(frozen=True)
class MCPToolResult:
    content: str
    is_error: bool = False


class OrbitMCPClient:
    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "OrbitMCPClient":
        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=["-m", "orbit.mcp_server"],
                    )
                )
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
        except Exception as exc:
            await stack.aclose()
            raise MCPError(f"Could not start Orbit MCP server: {exc}") from exc

        self._stack = stack
        self._session = session
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise MCPError("MCP client is not connected.")
        return self._session

    async def list_ollama_tools(self) -> list[dict[str, Any]]:
        try:
            result = await self._require_session().list_tools()
        except Exception as exc:
            raise MCPError(f"Could not list MCP tools: {exc}") from exc

        tools: list[dict[str, Any]] = []
        for tool in result.tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        try:
            result = await self._require_session().call_tool(name, arguments=arguments)
        except Exception as exc:
            raise MCPError(f"Tool '{name}' failed: {exc}") from exc

        text_parts: list[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                text_parts.append(text)

        if text_parts:
            content = "\n".join(text_parts)
        else:
            structured = getattr(result, "structuredContent", None)
            content = json.dumps(structured, ensure_ascii=False) if structured is not None else ""

        return MCPToolResult(content=content, is_error=result.isError is True)
