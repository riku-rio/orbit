from __future__ import annotations

from orbit.browser.chrome_devtools import ChromeDevToolsClient
from orbit.mcp_proxy import OrbitMCPServer
from orbit.mcp_tools import register_tools
from orbit.runtime import orbit_lifespan

browser_proxy = ChromeDevToolsClient.for_agent_browser()
mcp = OrbitMCPServer(
    "orbit",
    browser_proxy=browser_proxy,
    lifespan=orbit_lifespan,
)
register_tools(mcp)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
