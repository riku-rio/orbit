from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import orbit.browser.chrome_devtools as chrome_devtools
from orbit.browser.chrome_devtools import ChromeDevToolsClient


class TaskBoundContext:
    def __init__(self, value: object) -> None:
        self.value = value
        self.enter_task: asyncio.Task[object] | None = None
        self.exit_task: asyncio.Task[object] | None = None

    async def __aenter__(self) -> object:
        self.enter_task = asyncio.current_task()
        return self.value

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.exit_task = asyncio.current_task()
        if self.exit_task is not self.enter_task:
            raise RuntimeError(
                "Attempted to exit cancel scope in a different task than it was entered in"
            )


class FakeClientSession:
    def __init__(self, *_args: object) -> None:
        self.context = TaskBoundContext(self)
        self.initialize_task: asyncio.Task[object] | None = None

    async def __aenter__(self) -> "FakeClientSession":
        return await self.context.__aenter__()  # type: ignore[return-value]

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.context.__aexit__(exc_type, exc, tb)

    async def initialize(self) -> None:
        self.initialize_task = asyncio.current_task()

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(tools=[SimpleNamespace(name="navigate_page")])

    async def call_tool(self, name: str, arguments: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(content=[], structured_content=None, is_error=False)


class ChromeLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_child_context_closes_on_same_owner_task(self) -> None:
        transports: list[TaskBoundContext] = []
        sessions: list[FakeClientSession] = []

        def fake_stdio_client(*_args: object, **_kwargs: object) -> TaskBoundContext:
            context = TaskBoundContext((object(), object()))
            transports.append(context)
            return context

        def fake_session(*args: object) -> FakeClientSession:
            session = FakeClientSession(*args)
            sessions.append(session)
            return session

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = ChromeDevToolsClient(
                package="chrome-devtools-mcp@test",
                profile_path=root / "profile",
                stderr_path=root / "stderr.log",
            )
            with (
                patch.object(chrome_devtools, "stdio_client", fake_stdio_client),
                patch.object(chrome_devtools, "ClientSession", fake_session),
                patch.object(chrome_devtools.shutil, "which", return_value="npx"),
                patch.object(chrome_devtools, "BROWSER_LOG_DIR", root),
            ):
                # Orbit handles tool discovery, calls, and lifespan shutdown in
                # different tasks. The child MCP contexts must still be owned by
                # one long-lived runner task.
                tools = await asyncio.create_task(client.list_tools())
                await asyncio.create_task(
                    client.call_tool_raw(
                        "navigate_page",
                        {"url": "https://example.com"},
                    )
                )
                await asyncio.create_task(client.close())

        self.assertEqual([tool.name for tool in tools], ["navigate_page"])
        self.assertEqual(len(transports), 1)
        self.assertEqual(len(sessions), 1)
        self.assertIs(transports[0].enter_task, transports[0].exit_task)
        self.assertIs(sessions[0].context.enter_task, sessions[0].context.exit_task)
        self.assertIs(sessions[0].initialize_task, transports[0].enter_task)

    async def test_close_without_start_does_not_launch_child(self) -> None:
        client = ChromeDevToolsClient()
        await client.close()


if __name__ == "__main__":
    unittest.main()
