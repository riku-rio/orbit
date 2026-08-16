from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from orbit.browser.chrome_devtools import ChromeDevToolsClient


class OrbitMCPServer(MCPServer):
    """Orbit MCP server with Chrome DevTools exposed as transparent child tools."""

    def __init__(
        self,
        *args: Any,
        browser_proxy: ChromeDevToolsClient,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._browser_proxy = browser_proxy
        self._native_tool_names: frozenset[str] | None = None
        self._browser_tools: list[Any] | None = None
        self._browser_tool_names: frozenset[str] | None = None

    @property
    def browser_proxy(self) -> ChromeDevToolsClient:
        return self._browser_proxy

    async def _load_browser_tools(self) -> list[Any]:
        if self._browser_tools is None:
            tools = await self._browser_proxy.list_tools()
            self._browser_tools = tools
            self._browser_tool_names = frozenset(tool.name for tool in tools)
        return self._browser_tools

    async def list_tools(self) -> list[Any]:
        native_tools = await super().list_tools()
        browser_tools = await self._load_browser_tools()

        native_names = frozenset(tool.name for tool in native_tools)
        browser_names = self._browser_tool_names or frozenset()
        collisions = sorted(native_names & browser_names)
        if collisions:
            names = ", ".join(collisions)
            raise RuntimeError(
                f"Chrome DevTools tool name collision with Orbit tools: {names}"
            )

        self._native_tool_names = native_names
        return [*native_tools, *browser_tools]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        if self._native_tool_names is None:
            native_tools = await super().list_tools()
            self._native_tool_names = frozenset(tool.name for tool in native_tools)

        if name in self._native_tool_names:
            return await super().call_tool(name, arguments, context)

        if self._browser_tool_names is None:
            await self._load_browser_tools()
        if name in (self._browser_tool_names or frozenset()):
            return await self._browser_proxy.call_tool_raw(name, arguments)

        # Preserve MCPServer's normal unknown-tool behavior for names that belong
        # to neither Orbit nor Chrome DevTools.
        return await super().call_tool(name, arguments, context)
