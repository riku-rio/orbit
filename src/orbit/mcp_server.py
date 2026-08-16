from __future__ import annotations

from mcp.server import MCPServer

from orbit.mcp_tools import register_tools
from orbit.runtime import orbit_lifespan

mcp = MCPServer("orbit", lifespan=orbit_lifespan)
register_tools(mcp)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
