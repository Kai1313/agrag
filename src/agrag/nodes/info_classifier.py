"""Node 2: Retrieval Need Classifier (Do we need more info?)."""

import json
import logging
from typing import Any, Dict
from agrag.llm_client import llm_client
from agrag.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Information Need Classifier Agent in an enterprise Agentic RAG pipeline.
Your job is to determine whether the user's query requires external knowledge retrieval (from company Vector DB, Live Web Search, or Operational APIs) or can be answered purely with conversational/parametric knowledge.

Rules:
1. Return needs_retrieval = false for:
   - Conversational greetings, chit-chat, introductions (e.g. 'hello', 'how are you', 'who made you').
   - Elementary math and logic puzzles (e.g. 'what is 15 * 4', 'write a function to reverse a string').
   - General world knowledge definitions where company policies are irrelevant.
2. Return needs_retrieval = true for:
   - Company policies, IT rules, MFA requirements, session timeouts, remote work guidelines.
   - Specific employee status, Jira/incident tickets, credentials, system health.
   - Live external facts, regulatory standards (NIST, CSA), breaking news.
   - Any query requiring grounded verification.

Output MUST be valid JSON in this exact structure:
{
    "needs_retrieval": true | false,
    "reasoning": "<short explanation>"
}"""


async def info_classifier_node(state: AgentState) -> Dict[str, Any]:
    """Classifies whether the updated query needs external retrieval."""
    query = state.updated_query or state.user_query
    logger.info(f"Node 2 [Info Classifier] evaluating query: '{query}'")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query to evaluate:\n{query}"},
    ]

    response_text = await llm_client.chat_completion(
        messages=messages,
        model="deepseek-chat",
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    try:
        data = json.loads(response_text)
        needs_retrieval = bool(data.get("needs_retrieval", True))
    except Exception as e:
        logger.warning(f"Failed to parse Info Classifier JSON: {e}. Defaulting to needs_retrieval=True.")
        needs_retrieval = True

    return {"needs_retrieval": needs_retrieval}
