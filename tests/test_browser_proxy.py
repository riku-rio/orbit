from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_types import CallToolResult, ImageContent, TextContent, Tool as MCPTool

from orbit.browser.chrome_devtools import _server_command
from orbit.chat import _tool_message
from orbit.mcp_client import _server_environment, _tool_result_from_mcp
from orbit.mcp_proxy import OrbitMCPServer


class FakeBrowserProxy:
    def __init__(self, tools: list[MCPTool]) -> None:
        self.tools = tools
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    async def list_tools(self) -> list[MCPTool]:
        return self.tools

    async def call_tool_raw(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> CallToolResult:
        self.calls.append((name, arguments))
        return CallToolResult(
            content=[TextContent(type="text", text=f"browser:{name}")]
        )

    async def close(self) -> None:
        self.closed = True


def browser_tool(name: str = "navigate_page") -> MCPTool:
    return MCPTool(
        name=name,
        description="Browser tool",
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )


class BrowserProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_native_and_browser_tools_without_rewriting_schema(self):
        proxy = FakeBrowserProxy([browser_tool()])
        server = OrbitMCPServer("test", browser_proxy=proxy)

        @server.tool()
        def native_tool(value: str) -> str:
            return value

        tools = await server.list_tools()
        by_name = {tool.name: tool for tool in tools}

        self.assertEqual(set(by_name), {"native_tool", "navigate_page"})
        self.assertEqual(
            by_name["navigate_page"].input_schema,
            proxy.tools[0].input_schema,
        )

    async def test_routes_browser_tool_to_child_server(self):
        proxy = FakeBrowserProxy([browser_tool()])
        server = OrbitMCPServer("test", browser_proxy=proxy)
        await server.list_tools()

        result = await server.call_tool(
            "navigate_page",
            {"url": "https://example.com"},
        )

        self.assertEqual(
            proxy.calls,
            [("navigate_page", {"url": "https://example.com"})],
        )
        self.assertEqual(result.content[0].text, "browser:navigate_page")

    async def test_routes_native_tool_locally(self):
        proxy = FakeBrowserProxy([browser_tool()])
        server = OrbitMCPServer("test", browser_proxy=proxy)

        @server.tool()
        def echo(value: str) -> str:
            return f"native:{value}"

        result = await server.call_tool("echo", {"value": "hello"})

        self.assertEqual(proxy.calls, [])
        self.assertEqual(result.content[0].text, "native:hello")

    async def test_rejects_browser_native_name_collision(self):
        proxy = FakeBrowserProxy([browser_tool("echo")])
        server = OrbitMCPServer("test", browser_proxy=proxy)

        @server.tool()
        def echo(value: str) -> str:
            return value

        with self.assertRaisesRegex(RuntimeError, "name collision"):
            await server.list_tools()


class BrowserConfigurationTests(unittest.TestCase):
    def test_full_browser_command_exposes_current_tool_families(self):
        command, args = _server_command(
            "chrome-devtools-mcp@test",
            Path("/tmp/orbit-browser-profile"),
            headless=False,
            full_browser=True,
            enable_screencast=True,
        )
        rendered = " ".join([command, *args])

        self.assertNotIn("--headless", args)
        self.assertIn("--memory-debugging", args)
        self.assertIn("--category-extensions", args)
        self.assertIn("--category-pwa", args)
        self.assertIn("--category-experimental-third-party", args)
        self.assertIn("--category-experimental-webmcp", args)
        self.assertIn("--experimental-vision", args)
        self.assertIn("--experimental-screencast", args)
        self.assertIn("--allow-unrestricted-paths", args)
        self.assertIn("--chrome-arg=--enable-features=WebMCP", args)
        self.assertIn("--user-data-dir=/tmp/orbit-browser-profile", rendered)

    def test_orbit_mcp_subprocess_forwards_path_for_npx_discovery(self):
        with patch.dict(os.environ, {"PATH": "/test/bin"}, clear=False):
            env = _server_environment()
        self.assertEqual(env["PATH"], "/test/bin")


class BrowserMediaTests(unittest.TestCase):
    def test_image_content_is_forwarded_to_ollama_tool_message(self):
        raw = CallToolResult(
            content=[
                TextContent(type="text", text="Screenshot captured."),
                ImageContent(
                    type="image",
                    data="aGVsbG8=",
                    mime_type="image/png",
                ),
            ]
        )

        result = _tool_result_from_mcp(raw)
        message = _tool_message("take_screenshot", result.content, result.images)

        self.assertEqual(result.content, "Screenshot captured.")
        self.assertEqual(result.images, ("aGVsbG8=",))
        self.assertEqual(message["images"], ["aGVsbG8="])
        self.assertEqual(message["tool_name"], "take_screenshot")


if __name__ == "__main__":
    unittest.main()
