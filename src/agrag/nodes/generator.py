"""Node 4: DeepSeek Response Generator (Chat & Reasoner)."""

import logging
from typing import Any, Dict
from agrag.llm_client import llm_client
from agrag.state import AgentState

logger = logging.getLogger(__name__)

GROUNDED_SYSTEM_PROMPT = """You are an intelligent AI assistant for Acme Cloud Corp.
Your primary directive is to answer the user's question accurately, concisely, and completely.

Instructions:
1. Ground your response strictly on the provided Context.
2. If citing specific policies or operational records, include the source citation in brackets: e.g. [Source: <Source Title>].
3. If the context does not contain enough information to answer a part of the question, state honestly what is missing. Do NOT fabricate facts, dates, or policy details.
4. Maintain a helpful, professional, and clear tone."""

DIRECT_SYSTEM_PROMPT = """You are a helpful, intelligent AI assistant for Acme Cloud Corp.
Answer the user's query directly and conversationally."""


async def generator_node(state: AgentState) -> Dict[str, Any]:
    """Generates draft response using DeepSeek (deepseek-chat or deepseek-reasoner)."""
    query = state.updated_query or state.user_query
    model_name = state.model_name
    logger.info(f"Node 4 [Generator] synthesizing response using model '{model_name}'")

    if state.fused_context:
        system_content = GROUNDED_SYSTEM_PROMPT
        user_content = f"""Context:
{state.fused_context}

User Query:
{query}

Answer:"""
    else:
        system_content = DIRECT_SYSTEM_PROMPT
        user_content = query

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    response_text = await llm_client.chat_completion(
        messages=messages,
        model=model_name,
        temperature=0.2,
        max_tokens=2048,
    )

    return {"initial_response": response_text.strip()}
