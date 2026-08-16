from __future__ import annotations

import base64
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_LOG_DIR = Path.home() / ".orbit"
MCP_STDERR_PATH = MCP_LOG_DIR / "mcp-stderr.log"
MCP_STDERR_TAIL_LINES = 40


class MCPError(RuntimeError):
    pass


@dataclass(frozen=True)
class MCPToolResult:
    content: str
    images: tuple[str, ...] = ()
    is_error: bool = False


def _server_environment() -> dict[str, str]:
    """Forward Orbit/ML configuration plus executable discovery paths."""
    env = {
        "PYTHONFAULTHANDLER": "1",
        "PYTHONUNBUFFERED": "1",
        # Transformers v5 uses async tensor materialization by default. On the
        # Windows memory subprocess this can crash inside spawn_materialize
        # while loading large safetensor checkpoints. Prefer sequential loading
        # unless the user explicitly opts back into async loading.
        "HF_DEACTIVATE_ASYNC_LOAD": os.getenv("HF_DEACTIVATE_ASYNC_LOAD", "1"),
    }
    forwarded_names = {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "SENTENCE_TRANSFORMERS_HOME",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TORCH_HOME",
        "USERPROFILE",
        "WINDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_PATH",
        "CUDA_HOME",
        "PYTORCH_CUDA_ALLOC_CONF",
    }
    for name, value in os.environ.items():
        if (
            name.startswith("ORBIT_")
            or name.startswith("HF_")
            or name.startswith("TRANSFORMERS_")
            or name.upper() in forwarded_names
        ):
            env[name] = value
    return env


def _stderr_tail() -> str:
    try:
        text = MCP_STDERR_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-MCP_STDERR_TAIL_LINES:])


def _dump_content_item(item: Any) -> str:
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        try:
            payload = model_dump(mode="json", by_alias=True, exclude_none=True)
            return json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    return str(item)


def _tool_result_from_mcp(result: Any) -> MCPToolResult:
    content_parts: list[str] = []
    images: list[str] = []

    for item in result.content:
        item_type = getattr(item, "type", None)
        if item_type == "image":
            data = getattr(item, "data", None)
            if isinstance(data, str) and data:
                images.append(data)
                continue
            if isinstance(data, bytes) and data:
                images.append(base64.b64encode(data).decode("ascii"))
                continue

        text = getattr(item, "text", None)
        if isinstance(text, str):
            content_parts.append(text)
            continue

        # Chrome DevTools currently returns text/image content, but preserve any
        # future MCP content kinds as compact JSON instead of silently dropping them.
        content_parts.append(_dump_content_item(item))

    if content_parts:
        content = "\n".join(content_parts)
    else:
        structured = getattr(result, "structured_content", None)
        content = (
            json.dumps(structured, ensure_ascii=False)
            if structured is not None
            else ""
        )

    return MCPToolResult(
        content=content,
        images=tuple(images),
        is_error=result.is_error is True,
    )


class OrbitMCPClient:
    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "OrbitMCPClient":
        stack = AsyncExitStack()
        try:
            MCP_LOG_DIR.mkdir(parents=True, exist_ok=True)
            errlog = stack.enter_context(
                MCP_STDERR_PATH.open("w", encoding="utf-8", buffering=1)
            )
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=["-X", "faulthandler", "-u", "-m", "orbit.mcp_server"],
                        env=_server_environment(),
                    ),
                    errlog=errlog,
                )
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
        except Exception as exc:
            await stack.aclose()
            stderr = _stderr_tail()
            detail = f"\nMCP server stderr:\n{stderr}" if stderr else ""
            raise MCPError(f"Could not start Orbit MCP server: {exc}{detail}") from exc

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
            stderr = _stderr_tail()
            detail = f"\nMCP server stderr:\n{stderr}" if stderr else ""
            raise MCPError(f"Could not list MCP tools: {exc}{detail}") from exc

        tools: list[dict[str, Any]] = []
        for tool in result.tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
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
            stderr = _stderr_tail()
            detail = f"\nMCP server stderr:\n{stderr}" if stderr else ""
            raise MCPError(f"Tool '{name}' failed: {exc}{detail}") from exc

        return _tool_result_from_mcp(result)
