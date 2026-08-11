from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules

from mcp.server import MCPServer


def register_tools(mcp: MCPServer) -> None:
    """Discover and register every MCP tool module in this package."""
    for module_info in iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue

        module = import_module(f"{__name__}.{module_info.name}")
        register = getattr(module, "register", None)
        if not callable(register):
            raise RuntimeError(
                f"MCP tool module '{module.__name__}' must define register(mcp)."
            )
        register(mcp)
