from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbit.browser.chrome_devtools import (
    ChromeToolResult,
    _server_command,
    _server_environment,
)
from orbit.browser.search import WebSearchError, WebSearchService


class FakeChromeClient:
    def __init__(
        self,
        *,
        pages: list[dict[str, object]],
        payload: dict[str, object],
        fail_tool: str | None = None,
    ) -> None:
        self.pages = pages
        self.payload = payload
        self.fail_tool = fail_tool
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> ChromeToolResult:
        self.calls.append((name, arguments))
        if name == self.fail_tool:
            return ChromeToolResult(content="synthetic failure", is_error=True)
        if name == "list_pages":
            return ChromeToolResult(
                content="",
                structured_content={"pages": self.pages},
            )
        if name == "new_page":
            self.pages = [
                {
                    "id": 1,
                    "url": str(arguments["url"]),
                    "selected": True,
                }
            ]
            return ChromeToolResult(content="opened")
        if name == "evaluate_script":
            return ChromeToolResult(
                content=(
                    "Script ran on page and returned:\n"
                    "```json\n"
                    f"{json.dumps(self.payload)}\n"
                    "```"
                )
            )
        return ChromeToolResult(content="ok")


class ChromeDevToolsConfigTests(unittest.TestCase):
    def test_server_command_uses_headless_private_profile(self):
        command, args = _server_command(
            "chrome-devtools-mcp@test",
            Path("/tmp/orbit-search-profile"),
        )
        rendered = " ".join([command, *args])
        self.assertIn("chrome-devtools-mcp@test", rendered)
        self.assertIn("--headless", rendered)
        self.assertIn("--user-data-dir=/tmp/orbit-search-profile", rendered)
        self.assertIn("--experimental-structured-content", rendered)
        self.assertIn("--no-usage-statistics", rendered)
        self.assertIn("--no-performance-crux", rendered)

    def test_server_environment_disables_telemetry_and_update_checks(self):
        env = _server_environment()
        self.assertEqual(env["CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS"], "1")
        self.assertEqual(env["CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS"], "1")


class WebSearchServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_is_lazy_and_returns_only_first_five_results(self):
        payload = {
            "status": "ok",
            "results": [
                {
                    "title": f"Result {index}",
                    "source": f"example{index}.com",
                    "url": f"https://example{index}.com/page",
                    "snippet": f"Snippet {index}",
                }
                for index in range(1, 7)
            ],
        }
        client = FakeChromeClient(
            pages=[{"id": 1, "url": "about:blank", "selected": True}],
            payload=payload,
        )
        service = WebSearchService(client_factory=lambda: client)
        self.assertFalse(client.started)

        response = await service.search("  python   mcp sdk  ")

        self.assertTrue(client.started)
        self.assertEqual(response.query, "python mcp sdk")
        self.assertEqual(len(response.results), 5)
        self.assertEqual(response.results[0].rank, 1)
        self.assertEqual(response.results[-1].title, "Result 5")

        navigate_call = next(call for call in client.calls if call[0] == "navigate_page")
        query = parse_qs(urlparse(str(navigate_call[1]["url"])).query)
        self.assertEqual(query["q"], ["python mcp sdk"])
        self.assertEqual(query["num"], ["10"])

    async def test_search_prefers_existing_google_page_and_closes_other_tabs(self):
        client = FakeChromeClient(
            pages=[
                {"id": 1, "url": "https://example.com/", "selected": True},
                {"id": 2, "url": "https://www.google.com/", "selected": False},
                {"id": 3, "url": "about:blank", "selected": False},
            ],
            payload={"status": "ok", "results": []},
        )
        service = WebSearchService(client_factory=lambda: client)

        await service.search("orbit")

        self.assertIn(
            ("select_page", {"pageId": 2, "bringToFront": False}),
            client.calls,
        )
        closed_ids = {
            int(arguments["pageId"])
            for name, arguments in client.calls
            if name == "close_page"
        }
        self.assertEqual(closed_ids, {1, 3})

    async def test_search_creates_a_page_when_chrome_has_none(self):
        client = FakeChromeClient(
            pages=[],
            payload={"status": "ok", "results": []},
        )
        service = WebSearchService(client_factory=lambda: client)

        await service.search("orbit")

        self.assertTrue(any(name == "new_page" for name, _ in client.calls))

    async def test_google_block_is_reported_without_hiding_reason(self):
        client = FakeChromeClient(
            pages=[{"id": 1, "url": "https://www.google.com/", "selected": True}],
            payload={
                "status": "blocked",
                "reason": "Google presented an anti-bot or unusual-traffic page.",
                "results": [],
            },
        )
        service = WebSearchService(client_factory=lambda: client)

        with self.assertRaisesRegex(WebSearchError, "anti-bot"):
            await service.search("orbit")

    async def test_browser_tool_failure_resets_private_browser(self):
        client = FakeChromeClient(
            pages=[{"id": 1, "url": "about:blank", "selected": True}],
            payload={"status": "ok", "results": []},
            fail_tool="navigate_page",
        )
        service = WebSearchService(client_factory=lambda: client)

        with self.assertRaisesRegex(WebSearchError, "synthetic failure"):
            await service.search("orbit")

        self.assertTrue(client.closed)

    async def test_empty_query_does_not_start_browser(self):
        client = FakeChromeClient(
            pages=[{"id": 1, "url": "about:blank", "selected": True}],
            payload={"status": "ok", "results": []},
        )
        service = WebSearchService(client_factory=lambda: client)

        with self.assertRaisesRegex(WebSearchError, "non-empty query"):
            await service.search("   ")

        self.assertFalse(client.started)

    async def test_close_does_not_start_browser(self):
        created = 0

        def factory() -> FakeChromeClient:
            nonlocal created
            created += 1
            return FakeChromeClient(pages=[], payload={"status": "ok", "results": []})

        service = WebSearchService(client_factory=factory)
        await service.close()
        self.assertEqual(created, 0)


if __name__ == "__main__":
    unittest.main()
