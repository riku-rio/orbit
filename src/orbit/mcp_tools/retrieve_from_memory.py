from __future__ import annotations

import json

from mcp.server import MCPServer

from orbit.memory.runtime import get_memory_service


def register(mcp: MCPServer) -> None:
    @mcp.tool()
    def retrieve_from_memory(query: str, limit: int = 5) -> str:
        """Retrieve semantically relevant persistent memories with dense RAG + reranking.

        Use this when earlier preferences, facts, constraints, decisions, or project
        context may materially help with the current task. Formulate a semantic query
        for what you need; do not call this mechanically on every turn.
        """
        hits = get_memory_service().retrieve(query, limit=limit)
        return json.dumps(
            {
                "memories": [
                    {
                        "rank": hit.rank,
                        "id": hit.record.id,
                        "text": hit.record.text,
                        "stored_at": hit.record.stored_at,
                    }
                    for hit in hits
                ]
            },
            ensure_ascii=False,
        )
