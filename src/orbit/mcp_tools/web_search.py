from __future__ import annotations

import json

from mcp.server import MCPServer
from mcp.server.mcpserver import Context

from orbit.browser.search import WebSearchError
from orbit.runtime import OrbitRuntime


def register(mcp: MCPServer) -> None:
    @mcp.tool()
    async def web_search(query: str, ctx: Context[OrbitRuntime]) -> str:
        """Search the public web with Google and return compact organic results.

        Use this when current or external information is needed. Pass a specific search
        query. The tool returns up to five organic results with title, source, URL, and
        a best-effort snippet. It does not return the full search-results page.
        """
        try:
            result = await ctx.request_context.lifespan_context.web_search.search(query)
        except WebSearchError as exc:
            raise RuntimeError(str(exc)) from exc
        return json.dumps(result.as_dict(), ensure_ascii=False)
