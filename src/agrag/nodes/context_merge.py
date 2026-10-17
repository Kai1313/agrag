"""Node 3.5: Context Fusion & Provenance-Tagged Markdown Concatenation."""

import logging
from typing import Any, Dict
from agrag.state import AgentState

logger = logging.getLogger(__name__)


async def context_merge_node(state: AgentState) -> Dict[str, Any]:
    """Merges all retrieved chunks into a structured, provenance-tagged Markdown block."""
    chunks = state.retrieved_chunks
    logger.info(f"Node 3.5 [Context Merge] fusing {len(chunks)} retrieved chunks")

    if not chunks:
        return {"fused_context": None}

    formatted_sections = []
    for i, chunk in enumerate(chunks, 1):
        type_label = {
            "vector_db": "Internal Document",
            "web_search": "Live Web Search",
            "tool_api": "Operational Tool API",
        }.get(chunk.source_type, "External Source")

        header = f"### [{type_label}: {chunk.source_title}]"
        formatted_sections.append(f"{header}\n{chunk.content.strip()}")

    fused_context = "\n\n".join(formatted_sections)
    return {"fused_context": fused_context}
