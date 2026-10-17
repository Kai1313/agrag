"""Node 5: Response Relevance & Hallucination Evaluator Agent."""

import json
import logging
from typing import Any, Dict
from agrag.llm_client import llm_client
from agrag.state import AgentState, EvaluationVerdict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a rigorous Quality and Groundedness Evaluator Agent in an enterprise Agentic RAG pipeline.
Your task is to critique the synthesized response against the user query and the retrieved context.

Evaluation Dimensions:
1. Relevance: Does the response directly and adequately answer the user's question?
2. Groundedness (Hallucination check): Are all factual claims, dates, policies, and employee statuses strictly substantiated by the Context?
3. Information Absence check: Did the query ask for information that is completely absent from the Context?

Failure Type Rules:
- "none": The response is valid, relevant, and fully grounded.
- "missing_info": The Context simply does not contain the required facts or answer to the query (e.g. asking for unreleased plans or unknown records).
- "hallucination": The response makes ungrounded, fabricated, or contradictory claims that are unsupported by the Context.
- "unrelated": The response evaded the question or answered something completely different.

Output MUST be valid JSON in this exact structure:
{
    "is_valid": true | false,
    "failure_type": "none" | "missing_info" | "hallucination" | "unrelated",
    "critique": "<detailed diagnostic critique explaining your score and what to fix if retrying>"
}"""


async def response_evaluator_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates the draft response for relevance, groundedness, and hallucination."""
    query = state.updated_query or state.user_query
    response = state.initial_response or ""
    context = state.fused_context or "No external context retrieved (Direct generation path)."
    logger.info(f"Node 5 [Evaluator] checking draft response for query: '{query}'")

    user_prompt = f"""Context:
{context}

User Query:
{query}

Draft Response:
{response}

Evaluate the response now:"""

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
        is_valid = bool(data.get("is_valid", True))
        failure_type = data.get("failure_type", "none" if is_valid else "hallucination")
        if failure_type not in ["none", "missing_info", "hallucination", "unrelated"]:
            failure_type = "hallucination" if not is_valid else "none"
        critique = data.get("critique")
    except Exception as e:
        logger.warning(f"Failed to parse Evaluator JSON: {e}. Defaulting to valid.")
        is_valid = True
        failure_type = "none"
        critique = None

    verdict = EvaluationVerdict(
        is_valid=is_valid,
        failure_type=failure_type,
        critique=critique,
    )

    # Manage retry state
    new_retry_count = state.retry_count
    new_critique_history = list(state.critique_history)

    if not verdict.is_valid and verdict.failure_type in ["hallucination", "unrelated"]:
        new_retry_count += 1
        if verdict.critique:
            new_critique_history.append(verdict.critique)

    return {
        "evaluation": verdict,
        "retry_count": new_retry_count,
        "critique_history": new_critique_history,
    }
