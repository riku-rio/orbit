from __future__ import annotations

import asyncio
import os
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CHROME_DEVTOOLS_PACKAGE_ENV = "ORBIT_CHROME_DEVTOOLS_PACKAGE"
DEFAULT_CHROME_DEVTOOLS_PACKAGE = "chrome-devtools-mcp@latest"
BROWSER_LOG_DIR = Path.home() / ".orbit"
BROWSER_STDERR_PATH = BROWSER_LOG_DIR / "chrome-devtools-stderr.log"
BROWSER_PROFILE_PATH = BROWSER_LOG_DIR / "chrome-search-profile"
BROWSER_STDERR_TAIL_LINES = 40
DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0


class ChromeDevToolsError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChromeToolResult:
    content: str
    structured_content: Any | None = None
    is_error: bool = False


def _server_environment() -> dict[str, str]:
    """Pass only environment needed to discover Node, Chrome, and user paths."""
    allowed_names = {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
    }
    env = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in allowed_names
    }
    env["CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS"] = "1"
    env["CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS"] = "1"
    return env


def _server_command(
    package: str,
    profile_path: Path,
) -> tuple[str, list[str]]:
    npx_args = [
        "-y",
        package,
        "--headless",
        f"--user-data-dir={profile_path}",
        "--experimental-structured-content",
        "--no-usage-statistics",
        "--no-performance-crux",
        "--no-category-emulation",
        "--no-category-performance",
        "--no-category-network",
    ]
    if os.name == "nt":
        return "cmd", ["/c", "npx", *npx_args]
    return "npx", npx_args


def _stderr_tail() -> str:
    try:
        text = BROWSER_STDERR_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-BROWSER_STDERR_TAIL_LINES:])


class ChromeDevToolsClient:
    """Lazy stdio client for a private Chrome DevTools MCP child server."""

    def __init__(
        self,
        *,
        package: str | None = None,
        profile_path: Path | None = None,
        tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self._package = package or os.getenv(
            CHROME_DEVTOOLS_PACKAGE_ENV,
            DEFAULT_CHROME_DEVTOOLS_PACKAGE,
        )
        self._profile_path = profile_path or BROWSER_PROFILE_PATH
        self._tool_timeout_seconds = tool_timeout_seconds
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        if self._session is not None:
            return

        if shutil.which("npx") is None:
            raise ChromeDevToolsError(
                "Chrome DevTools MCP requires Node.js/npm (npx was not found in PATH)."
            )

        stack = AsyncExitStack()
        try:
            BROWSER_LOG_DIR.mkdir(parents=True, exist_ok=True)
            self._profile_path.mkdir(parents=True, exist_ok=True)
            errlog = stack.enter_context(
                BROWSER_STDERR_PATH.open("w", encoding="utf-8", buffering=1)
            )
            command, args = _server_command(self._package, self._profile_path)
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=command,
                        args=args,
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
            detail = f"\nChrome DevTools MCP stderr:\n{stderr}" if stderr else ""
            raise ChromeDevToolsError(
                "Could not start Chrome DevTools MCP. "
                "Install a supported Node.js/npm version and Google Chrome, then retry. "
                f"Underlying error: {exc}{detail}"
            ) from exc

        self._stack = stack
        self._session = session

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise ChromeDevToolsError("Chrome DevTools MCP is not connected.")
        return self._session

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ChromeToolResult:
        try:
            result = await asyncio.wait_for(
                self._require_session().call_tool(name, arguments=arguments),
                timeout=self._tool_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise ChromeDevToolsError(
                f"Chrome DevTools tool '{name}' timed out after "
                f"{self._tool_timeout_seconds:g}s."
            ) from exc
        except Exception as exc:
            stderr = _stderr_tail()
            detail = f"\nChrome DevTools MCP stderr:\n{stderr}" if stderr else ""
            raise ChromeDevToolsError(
                f"Chrome DevTools tool '{name}' failed: {exc}{detail}"
            ) from exc

        text_parts: list[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                text_parts.append(text)

        return ChromeToolResult(
            content="\n".join(text_parts),
            structured_content=getattr(result, "structured_content", None),
            is_error=result.is_error is True,
        )
