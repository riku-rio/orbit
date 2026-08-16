from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp.server import MCPServer

from orbit.browser.search import WebSearchService


@dataclass
class OrbitRuntime:
    web_search: WebSearchService


@asynccontextmanager
async def orbit_lifespan(server: MCPServer) -> AsyncIterator[OrbitRuntime]:
    # web_search intentionally owns a separate Chrome DevTools child/profile.
    # Its tab cleanup must never touch the agent's general-purpose browser.
    web_search = WebSearchService()
    try:
        yield OrbitRuntime(web_search=web_search)
    finally:
        await web_search.close()
        browser_proxy: Any | None = getattr(server, "browser_proxy", None)
        if browser_proxy is not None:
            await browser_proxy.close()
