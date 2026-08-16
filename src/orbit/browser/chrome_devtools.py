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
BROWSER_HEADLESS_ENV = "ORBIT_BROWSER_HEADLESS"
DEFAULT_CHROME_DEVTOOLS_PACKAGE = "chrome-devtools-mcp@1.7.0"
BROWSER_LOG_DIR = Path.home() / ".orbit"
SEARCH_STDERR_PATH = BROWSER_LOG_DIR / "chrome-search-stderr.log"
SEARCH_PROFILE_PATH = BROWSER_LOG_DIR / "chrome-search-profile"
AGENT_BROWSER_STDERR_PATH = BROWSER_LOG_DIR / "chrome-browser-stderr.log"
AGENT_BROWSER_PROFILE_PATH = BROWSER_LOG_DIR / "chrome-browser-profile"
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
        "DISPLAY",
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
        "WAYLAND_DISPLAY",
        "WINDIR",
        "XAUTHORITY",
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


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _server_command(
    package: str,
    profile_path: Path,
    *,
    headless: bool = True,
    full_browser: bool = False,
    enable_screencast: bool = False,
) -> tuple[str, list[str]]:
    npx_args = ["-y", package]
    if headless:
        npx_args.append("--headless")
    npx_args.extend(
        [
            f"--user-data-dir={profile_path}",
            "--experimental-structured-content",
            "--no-usage-statistics",
            "--no-performance-crux",
        ]
    )

    if full_browser:
        # The browser proxy is trusted local infrastructure. Enable the optional
        # Chrome DevTools tool families so Orbit exposes the complete current tool
        # surface rather than an arbitrary subset.
        npx_args.extend(
            [
                "--memory-debugging",
                "--category-extensions",
                "--category-pwa",
                "--category-experimental-third-party",
                "--category-experimental-webmcp",
                "--experimental-vision",
                "--allow-unrestricted-paths",
                "--chrome-arg=--enable-features=WebMCP",
            ]
        )
        if enable_screencast:
            npx_args.append("--experimental-screencast")
    else:
        # web_search only needs navigation + evaluate_script. Keep its child small
        # and completely separate from the agent's general-purpose browser.
        npx_args.extend(
            [
                "--no-category-emulation",
                "--no-category-performance",
                "--no-category-network",
            ]
        )

    if os.name == "nt":
        return "cmd", ["/c", "npx", *npx_args]
    return "npx", npx_args


def _stderr_tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-BROWSER_STDERR_TAIL_LINES:])


class ChromeDevToolsClient:
    """Lazy stdio client for one isolated Chrome DevTools MCP child server."""

    def __init__(
        self,
        *,
        package: str | None = None,
        profile_path: Path | None = None,
        stderr_path: Path | None = None,
        headless: bool = True,
        full_browser: bool = False,
        tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self._package = package or os.getenv(
            CHROME_DEVTOOLS_PACKAGE_ENV,
            DEFAULT_CHROME_DEVTOOLS_PACKAGE,
        )
        self._profile_path = profile_path or SEARCH_PROFILE_PATH
        self._stderr_path = stderr_path or SEARCH_STDERR_PATH
        self._headless = headless
        self._full_browser = full_browser
        self._tool_timeout_seconds = tool_timeout_seconds
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._start_lock = asyncio.Lock()

    @classmethod
    def for_agent_browser(cls) -> "ChromeDevToolsClient":
        return cls(
            profile_path=AGENT_BROWSER_PROFILE_PATH,
            stderr_path=AGENT_BROWSER_STDERR_PATH,
            headless=_env_flag(BROWSER_HEADLESS_ENV, default=False),
            full_browser=True,
        )

    async def start(self) -> None:
        if self._session is not None:
            return

        async with self._start_lock:
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
                    self._stderr_path.open("w", encoding="utf-8", buffering=1)
                )
                command, args = _server_command(
                    self._package,
                    self._profile_path,
                    headless=self._headless,
                    full_browser=self._full_browser,
                    enable_screencast=(
                        self._full_browser and shutil.which("ffmpeg") is not None
                    ),
                )
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
                stderr = _stderr_tail(self._stderr_path)
                detail = (
                    f"\nChrome DevTools MCP stderr:\n{stderr}" if stderr else ""
                )
                raise ChromeDevToolsError(
                    "Could not start Chrome DevTools MCP. "
                    "Install Node.js/npm and Google Chrome, then retry. "
                    f"Underlying error: {exc}{detail}"
                ) from exc

            self._stack = stack
            self._session = session

    async def close(self) -> None:
        async with self._start_lock:
            if self._stack is not None:
                await self._stack.aclose()
            self._stack = None
            self._session = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise ChromeDevToolsError("Chrome DevTools MCP is not connected.")
        return self._session

    async def list_tools(self) -> list[Any]:
        await self.start()
        try:
            result = await asyncio.wait_for(
                self._require_session().list_tools(),
                timeout=self._tool_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            await self.close()
            raise ChromeDevToolsError(
                "Chrome DevTools tool discovery timed out after "
                f"{self._tool_timeout_seconds:g}s."
            ) from exc
        except Exception as exc:
            stderr = _stderr_tail(self._stderr_path)
            await self.close()
            detail = f"\nChrome DevTools MCP stderr:\n{stderr}" if stderr else ""
            raise ChromeDevToolsError(
                f"Could not list Chrome DevTools tools: {exc}{detail}"
            ) from exc
        return list(result.tools)

    async def call_tool_raw(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        await self.start()
        try:
            return await asyncio.wait_for(
                self._require_session().call_tool(name, arguments=arguments),
                timeout=self._tool_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            await self.close()
            raise ChromeDevToolsError(
                f"Chrome DevTools tool '{name}' timed out after "
                f"{self._tool_timeout_seconds:g}s."
            ) from exc
        except Exception as exc:
            stderr = _stderr_tail(self._stderr_path)
            await self.close()
            detail = f"\nChrome DevTools MCP stderr:\n{stderr}" if stderr else ""
            raise ChromeDevToolsError(
                f"Chrome DevTools tool '{name}' failed: {exc}{detail}"
            ) from exc

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ChromeToolResult:
        result = await self.call_tool_raw(name, arguments)

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
