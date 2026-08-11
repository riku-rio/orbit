from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from orbit.mcp_tools import register_tools

mcp = FastMCP("orbit")
register_tools(mcp)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
