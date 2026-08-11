from __future__ import annotations

from datetime import datetime

from mcp.server.fastmcp import FastMCP


def _format_time(now: datetime) -> str:
    return now.strftime("%I:%M %p").lstrip("0")


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    def get_current_time() -> str:
        """Get the current local time in 12-hour format."""
        return _format_time(datetime.now())
