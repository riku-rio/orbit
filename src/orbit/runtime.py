from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server import MCPServer

from orbit.browser.search import WebSearchService


@dataclass
class OrbitRuntime:
    web_search: WebSearchService


@asynccontextmanager
async def orbit_lifespan(_server: MCPServer) -> AsyncIterator[OrbitRuntime]:
    web_search = WebSearchService()
    try:
        yield OrbitRuntime(web_search=web_search)
    finally:
        await web_search.close()
