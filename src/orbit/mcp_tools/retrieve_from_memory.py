from __future__ import annotations

import json

from mcp.server import MCPServer

from orbit.memory.runtime import get_memory_service


def register(mcp: MCPServer) -> None:
    @mcp.tool()
    def retrieve_from_memory(query: str, limit: int = 5) -> str:
        """Retrieve semantically relevant persistent memories with dense RAG + reranking.

        Call this proactively before answering when the request depends on information
        that may have been stored in previous sessions and is not already available in
        the current conversation. Strong examples include user identity/profile facts,
        preferences, prior project decisions/configuration, recurring commands/workflows,
        "what did I tell you...", "what do I prefer...", "what's my name?", and similar
        references to prior context. Questions about recurring use of this Orbit harness
        are also retrieval candidates when the answer may have been taught previously.

        Before claiming that you do not know a user-specific or project-specific fact,
        or asking the user to repeat it, try this tool once unless the current conversation
        already contains the answer. Do not call this mechanically for standalone general
        knowledge or generic tasks such as arithmetic, definitions, or "How do I center a div?".
        Formulate a semantic query describing the fact or context you need.
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
