from __future__ import annotations

import json

from mcp.server import MCPServer

from orbit.memory.runtime import get_memory_service


def register(mcp: MCPServer) -> None:
    @mcp.tool()
    def add_to_memory(memory: str) -> str:
        """Store one durable, self-contained long-term memory.

        Use this for stable user preferences, facts, project decisions, constraints,
        or important long-lived state that is likely to matter in future interactions.
        Do not store every message, transient chatter, credentials, or secrets.
        """
        result = get_memory_service().add(memory)
        return json.dumps(
            {
                "status": "already_stored" if result.already_stored else "stored",
                "memory": {
                    "id": result.record.id,
                    "text": result.record.text,
                    "stored_at": result.record.stored_at,
                },
            },
            ensure_ascii=False,
        )
