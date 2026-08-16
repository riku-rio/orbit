from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbit.browser.chrome_devtools import ChromeToolResult
from orbit.browser.search import WebSearchService


class StrictChromeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> ChromeToolResult:
        self.calls.append((name, arguments))
        if name == "list_pages":
            return ChromeToolResult(
                content="",
                structured_content={
                    "pages": [
                        {
                            "id": 1,
                            "url": "https://www.google.com/",
                            "selected": True,
                        }
                    ]
                },
            )
        if name == "evaluate_script":
            allowed = {"function", "args", "filePath", "dialogAction"}
            unknown = set(arguments) - allowed
            if unknown:
                raise AssertionError(f"unsupported evaluate_script arguments: {sorted(unknown)}")
            return ChromeToolResult(
                content=(
                    "Script ran on page and returned:\n"
                    "```json\n"
                    f"{json.dumps({'status': 'ok', 'results': []})}\n"
                    "```"
                )
            )
        return ChromeToolResult(content="ok")


class WebSearchSchemaCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluate_script_uses_only_pinned_1_7_arguments(self):
        client = StrictChromeClient()
        service = WebSearchService(client_factory=lambda: client)

        response = await service.search("orbit")

        self.assertEqual(response.results, ())
        evaluate_call = next(call for call in client.calls if call[0] == "evaluate_script")
        self.assertEqual(set(evaluate_call[1]), {"function"})


if __name__ == "__main__":
    unittest.main()
