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


@dataclass
class _ChromeRequest:
    kind: str
    future: asyncio.Future[Any]
    name: str | None = None
    arguments: dict[str, Any] | None = None


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
    """Lazy stdio client owned by one long-lived asyncio task.

    AnyIO's stdio transport opens a task-group cancel scope that must be exited
    by the same task that entered it. Orbit receives MCP requests and lifespan
    shutdown in different tasks, so keeping AsyncExitStack on the caller task
    causes ``Attempted to exit cancel scope in a different task`` on shutdown.

    This client therefore owns the entire child MCP lifecycle inside one runner
    task. Public methods only enqueue work and await a Future, so start/call/close
    are safe when invoked by different Orbit request tasks.
    """

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

        self._state_lock = asyncio.Lock()
        self._runner_task: asyncio.Task[None] | None = None
        self._requests: asyncio.Queue[_ChromeRequest] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._close_waiter: asyncio.Future[Any] | None = None

    @classmethod
    def for_agent_browser(cls) -> "ChromeDevToolsClient":
        return cls(
            profile_path=AGENT_BROWSER_PROFILE_PATH,
            stderr_path=AGENT_BROWSER_STDERR_PATH,
            headless=_env_flag(BROWSER_HEADLESS_ENV, default=False),
            full_browser=True,
        )

    def _startup_error(self, exc: BaseException) -> ChromeDevToolsError:
        stderr = _stderr_tail(self._stderr_path)
        detail = f"\nChrome DevTools MCP stderr:\n{stderr}" if stderr else ""
        return ChromeDevToolsError(
            "Could not start Chrome DevTools MCP. "
            "Install Node.js/npm and Google Chrome, then retry. "
            f"Underlying error: {exc}{detail}"
        )

    def _request_error(
        self,
        request: _ChromeRequest,
        exc: BaseException,
    ) -> ChromeDevToolsError:
        if isinstance(exc, asyncio.TimeoutError):
            if request.kind == "list_tools":
                return ChromeDevToolsError(
                    "Chrome DevTools tool discovery timed out after "
                    f"{self._tool_timeout_seconds:g}s."
                )
            return ChromeDevToolsError(
                f"Chrome DevTools tool '{request.name}' timed out after "
                f"{self._tool_timeout_seconds:g}s."
            )

        stderr = _stderr_tail(self._stderr_path)
        detail = f"\nChrome DevTools MCP stderr:\n{stderr}" if stderr else ""
        if request.kind == "list_tools":
            return ChromeDevToolsError(
                f"Could not list Chrome DevTools tools: {exc}{detail}"
            )
        return ChromeDevToolsError(
            f"Chrome DevTools tool '{request.name}' failed: {exc}{detail}"
        )

    @staticmethod
    def _set_result(future: asyncio.Future[Any], value: Any) -> None:
        if not future.done():
            future.set_result(value)

    @staticmethod
    def _set_exception(
        future: asyncio.Future[Any],
        error: BaseException,
    ) -> None:
        if not future.done():
            future.set_exception(error)

    def _fail_pending(
        self,
        requests: asyncio.Queue[_ChromeRequest],
        error: BaseException,
    ) -> None:
        while True:
            try:
                request = requests.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._set_exception(request.future, error)

    async def _run(
        self,
        requests: asyncio.Queue[_ChromeRequest],
        ready: asyncio.Future[None],
    ) -> None:
        stack = AsyncExitStack()
        close_request: _ChromeRequest | None = None
        failure: BaseException | None = None
        try:
            if shutil.which("npx") is None:
                raise ChromeDevToolsError(
                    "Chrome DevTools MCP requires Node.js/npm (npx was not found in PATH)."
                )

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
            self._set_result(ready, None)

            while True:
                request = await requests.get()
                if request.kind == "close":
                    close_request = request
                    break

                try:
                    if request.kind == "list_tools":
                        result = await asyncio.wait_for(
                            session.list_tools(),
                            timeout=self._tool_timeout_seconds,
                        )
                    elif request.kind == "call_tool":
                        result = await asyncio.wait_for(
                            session.call_tool(
                                request.name or "",
                                arguments=request.arguments or {},
                            ),
                            timeout=self._tool_timeout_seconds,
                        )
                    else:  # pragma: no cover - internal invariant
                        raise RuntimeError(f"Unknown Chrome request: {request.kind}")
                except Exception as exc:
                    failure = self._request_error(request, exc)
                    self._set_exception(request.future, failure)
                    break
                else:
                    self._set_result(request.future, result)
        except asyncio.CancelledError:
            failure = ChromeDevToolsError("Chrome DevTools MCP runner was cancelled.")
            if not ready.done():
                ready.cancel()
            raise
        except Exception as exc:
            if isinstance(exc, ChromeDevToolsError):
                failure = exc
            else:
                failure = self._startup_error(exc)
            if not ready.done():
                self._set_exception(ready, failure)
        finally:
            if failure is not None:
                self._fail_pending(requests, failure)

            cleanup_error: BaseException | None = None
            try:
                # Critical: all AsyncExitStack callbacks run in this same owner
                # task, matching the task that entered stdio_client/ClientSession.
                await stack.aclose()
            except Exception as exc:  # pragma: no cover - defensive cleanup path
                cleanup_error = ChromeDevToolsError(
                    f"Could not close Chrome DevTools MCP cleanly: {exc}"
                )

            if close_request is not None:
                if cleanup_error is None:
                    self._set_result(close_request.future, None)
                else:
                    self._set_exception(close_request.future, cleanup_error)
            elif cleanup_error is not None and failure is None:
                self._fail_pending(requests, cleanup_error)

    async def start(self) -> None:
        async with self._state_lock:
            task = self._runner_task
            if task is None or task.done():
                loop = asyncio.get_running_loop()
                requests: asyncio.Queue[_ChromeRequest] = asyncio.Queue()
                ready: asyncio.Future[None] = loop.create_future()
                task = asyncio.create_task(
                    self._run(requests, ready),
                    name="orbit-chrome-devtools",
                )
                self._runner_task = task
                self._requests = requests
                self._ready = ready
                self._close_waiter = None
            ready = self._ready

        if ready is None:  # pragma: no cover - internal invariant
            raise ChromeDevToolsError("Chrome DevTools MCP runner did not initialize.")
        await ready

    async def _submit(
        self,
        kind: str,
        *,
        name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        async with self._state_lock:
            task = self._runner_task
            requests = self._requests
            closing = self._close_waiter is not None
            if task is None or task.done() or requests is None or closing:
                raise ChromeDevToolsError("Chrome DevTools MCP is not connected.")
            requests.put_nowait(
                _ChromeRequest(
                    kind=kind,
                    future=future,
                    name=name,
                    arguments=arguments,
                )
            )

        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def close(self) -> None:
        async with self._state_lock:
            task = self._runner_task
            requests = self._requests
            if task is None:
                return
            if task.done():
                self._runner_task = None
                self._requests = None
                self._ready = None
                self._close_waiter = None
                return

            close_waiter = self._close_waiter
            if close_waiter is None:
                if requests is None:  # pragma: no cover - internal invariant
                    return
                close_waiter = asyncio.get_running_loop().create_future()
                self._close_waiter = close_waiter
                requests.put_nowait(
                    _ChromeRequest(kind="close", future=close_waiter)
                )

        try:
            await close_waiter
            await asyncio.shield(task)
        finally:
            async with self._state_lock:
                if self._runner_task is task and task.done():
                    self._runner_task = None
                    self._requests = None
                    self._ready = None
                    self._close_waiter = None

    async def list_tools(self) -> list[Any]:
        result = await self._submit("list_tools")
        return list(result.tools)

    async def call_tool_raw(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        return await self._submit(
            "call_tool",
            name=name,
            arguments=arguments,
        )

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
