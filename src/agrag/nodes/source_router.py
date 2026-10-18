"""Node 3: Source Selector & Concurrent Retrieval Router."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from agrag.llm_client import llm_client
from agrag.state import AgentState, RetrievedChunk, SourceType
from agrag.tools.custom_tools import tool_registry
from agrag.tools.search import WebSearchRetriever
from agrag.tools.vector_store import ChromaVectorRetriever

logger = logging.getLogger(__name__)

# Lazy singleton holders
_vector_retriever: Optional[ChromaVectorRetriever] = None
_web_retriever: Optional[WebSearchRetriever] = None


def get_vector_retriever() -> ChromaVectorRetriever:
    """Lazily initialize and return the ChromaVectorRetriever singleton."""
    global _vector_retriever
    if _vector_retriever is None:
        _vector_retriever = ChromaVectorRetriever()
    return _vector_retriever


def get_web_retriever() -> WebSearchRetriever:
    """Lazily initialize and return the WebSearchRetriever singleton."""
    global _web_retriever
    if _web_retriever is None:
        _web_retriever = WebSearchRetriever()
    return _web_retriever

SYSTEM_PROMPT = """You are a Source Selector Agent in an enterprise Agentic RAG pipeline.
Your job is to route the query to the appropriate knowledge silos and specify tool parameters if needed.

Available Sources:
1. "vector_db": Acme Cloud Corp internal IT security handbook, policies, data retention rules, session timeouts.
2. "web_search": Up-to-date public information, industry security frameworks (NIST, CSA), news.
3. "tool_api": Live operational tools:
{tools_description}

Rules:
- You may select 1, 2, or all 3 sources depending on what the query needs.
- If "tool_api" is selected, provide the tool name and arguments in the "tool_calls" array.
- Output MUST be valid JSON in this exact structure:
{{
    "selected_sources": ["vector_db" | "web_search" | "tool_api"],
    "tool_calls": [
        {{"tool_name": "<name>", "arguments": {{...}}}}
    ],
    "reasoning": "<short explanation>"
}}"""


async def source_router_node(state: AgentState) -> Dict[str, Any]:
    """Selects appropriate knowledge silos and dispatches concurrent retrieval tasks."""
    query = state.updated_query or state.user_query
    logger.info(f"Node 3 [Source Router] routing query: '{query}'")

    tools_desc = tool_registry.get_tools_description_prompt()
    formatted_system_prompt = SYSTEM_PROMPT.format(tools_description=tools_desc)

    messages = [
        {"role": "system", "content": formatted_system_prompt},
        {"role": "user", "content": f"Query:\n{query}"},
    ]

    response_text = await llm_client.chat_completion(
        messages=messages,
        model="deepseek-chat",
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    try:
        data = json.loads(response_text)
        raw_sources = data.get("selected_sources", ["vector_db"])
        tool_calls = data.get("tool_calls", [])
    except Exception as e:
        logger.warning(f"Failed to parse Source Router JSON: {e}. Defaulting to vector_db.")
        raw_sources = ["vector_db"]
        tool_calls = []

    # Filter to valid SourceType literals
    valid_source_set = {"vector_db", "web_search", "tool_api"}
    selected_sources: List[SourceType] = [
        s for s in raw_sources if s in valid_source_set
    ] or ["vector_db"]

    # Dispatch concurrent retrieval tasks
    tasks = []

    # 1. Vector DB retrieval
    if "vector_db" in selected_sources:
        tasks.append(get_vector_retriever().retrieve(query, top_k=3))

    # 2. Web Search retrieval
    if "web_search" in selected_sources:
        tasks.append(get_web_retriever().retrieve(query, top_k=2))


    # 3. Tool API execution
    if "tool_api" in selected_sources:
        for call in tool_calls:
            t_name = call.get("tool_name")
            t_args = call.get("arguments", {})
            tool = tool_registry.get_tool(t_name)
            if tool:
                tasks.append(_execute_tool_wrapper(tool, t_args))

    # Execute all retrieval tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    collected_chunks: List[RetrievedChunk] = []
    for res in results:
        if isinstance(res, Exception):
            logger.error(f"Retrieval task failed with exception: {res}")
            continue
        if isinstance(res, list):
            collected_chunks.extend(res)
        elif isinstance(res, RetrievedChunk):
            collected_chunks.append(res)

    return {
        "selected_sources": selected_sources,
        "retrieved_chunks": collected_chunks,
    }


async def _execute_tool_wrapper(tool, args: Dict[str, Any]) -> RetrievedChunk:
    """Helper to execute an operational tool safely."""
    try:
        return await tool.execute(**args)
    except Exception as e:
        logger.error(f"Tool {tool.name} execution failed: {e}")
        return RetrievedChunk(
            source_type="tool_api",
            source_title=f"Tool Execution Error ({tool.name})",
            content=f'{{"error": "Failed to execute tool: {str(e)}"}}',
            metadata={"tool_name": tool.name},
            score=0.0,
        )
