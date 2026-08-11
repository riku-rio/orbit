from __future__ import annotations

import json

from mcp.server import MCPServer

from orbit.memory.runtime import get_memory_service


def register(mcp: MCPServer) -> None:
    @mcp.tool()
    def add_to_memory(memory: str) -> str:
        """Store one durable, self-contained long-term memory.

        Call this proactively when the user reveals stable information that is likely
        to matter in future sessions, even if they did not explicitly ask you to
        remember it. Strong examples include identity/profile facts ("My name is ..."),
        stable preferences, persistent project decisions/configuration, constraints,
        and recurring commands or workflows. Explicit "remember" or "memorize" requests
        are also strong triggers when the content is safe to store.

        Do not store every message, transient chatter, temporary moods, generic factual
        knowledge, tool output, credentials, passwords, tokens, or secrets. Prefer one
        concise, self-contained durable fact per call.
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
