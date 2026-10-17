"""Node 1: History-Aware Query Rewriter & Keyword Expander."""

import json
import logging
from typing import Any, Dict
from agrag.llm_client import llm_client
from agrag.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Query Rewriter Agent in an Agentic RAG pipeline.
Your task is to rewrite the user's latest query into an unambiguous, keyword-rich, standalone search query.

Guidelines:
1. Resolve ambiguous pronouns and coreferences (e.g. 'it', 'they', 'its policy', 'his account') using the provided conversation history.
2. If previous evaluation critique is present, strictly address the missing information or hallucination issues flagged in the critique.
3. Keep the query focused and search-friendly.
4. Output MUST be valid JSON in this exact structure:
{
    "updated_query": "<your rewritten query>"
}"""


async def query_rewriter_node(state: AgentState) -> Dict[str, Any]:
    """Rewrites and disambiguates the user's query taking history and critiques into account."""
    logger.info(f"Node 1 [Query Rewriter] processing query: '{state.user_query}' (retry_count: {state.retry_count})")

    # Format history and critique if available
    history_str = "\n".join(
        f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
        for m in state.messages_history
    ) or "None (First turn in conversation)"

    critique_str = (
        state.critique_history[-1]
        if state.critique_history
        else "None (Initial attempt)"
    )

    user_prompt = f"""Conversation History:
{history_str}

Previous Evaluation Critique:
{critique_str}

Original User Query:
{state.user_query}

Rewrite the query now:"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response_text = await llm_client.chat_completion(
        messages=messages,
        model="deepseek-chat",
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    try:
        data = json.loads(response_text)
        updated_query = data.get("updated_query", state.user_query)
    except Exception as e:
        logger.warning(f"Failed to parse Query Rewriter JSON response: {e}. Using raw text.")
        updated_query = response_text.strip().strip('"') or state.user_query

    return {"updated_query": updated_query}
