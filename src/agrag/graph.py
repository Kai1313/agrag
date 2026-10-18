"""LangGraph Cyclic State Machine Assembly & Execution Control."""

import logging
from typing import Any, Dict, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agrag.nodes.context_merge import context_merge_node
from agrag.nodes.evaluator import response_evaluator_node
from agrag.nodes.generator import generator_node
from agrag.nodes.info_classifier import info_classifier_node
from agrag.nodes.rewriter import query_rewriter_node
from agrag.nodes.source_router import source_router_node
from agrag.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal Completion & Fallback Nodes
# ---------------------------------------------------------------------------


async def final_success_node(state: AgentState) -> Dict[str, Any]:
    """Terminal node for verified, grounded responses."""
    logger.info("Graph: Final Success reached.")
    updated_history = list(state.messages_history)
    updated_history.append({"role": "user", "content": state.user_query})
    updated_history.append(
        {"role": "assistant", "content": state.initial_response or ""}
    )
    return {
        "final_response": state.initial_response,
        "status": "completed",
        "messages_history": updated_history,
    }


async def graceful_fallback_node(state: AgentState) -> Dict[str, Any]:
    """Terminal node for missing information (fast-abort path)."""
    logger.info("Graph: Graceful Fallback reached (missing information).")
    critique = (
        state.evaluation.critique
        if state.evaluation and state.evaluation.critique
        else "The requested information could not be verified in any internal or external knowledge source."
    )
    fallback_message = (
        f"I searched our internal knowledge base, operational tools, and external sources, but could not verify "
        f"authoritative information to answer your question: '{state.user_query}'.\n\n"
        f"**Notice**: {critique}\n\n"
        f"If you believe this information should exist, please contact the IT Helpdesk or check Jira/ServiceNow tickets."
    )
    updated_history = list(state.messages_history)
    updated_history.append({"role": "user", "content": state.user_query})
    updated_history.append({"role": "assistant", "content": fallback_message})
    return {
        "final_response": fallback_message,
        "status": "fallback",
        "messages_history": updated_history,
    }


async def capped_fallback_node(state: AgentState) -> Dict[str, Any]:
    """Terminal node when maximum retry limit is reached without validation."""
    logger.info("Graph: Capped Fallback reached (max retries exceeded).")
    critique = (
        state.evaluation.critique
        if state.evaluation and state.evaluation.critique
        else "Validation attempts could not verify groundedness."
    )
    warning_message = (
        f"> ⚠️ **Warning**: This response could not be fully verified against authoritative sources after "
        f"{state.retry_count} validation attempts.\n\n"
        f"{state.initial_response or ''}\n\n"
        f"*(Validation Note: {critique})*"
    )
    updated_history = list(state.messages_history)
    updated_history.append({"role": "user", "content": state.user_query})
    updated_history.append({"role": "assistant", "content": warning_message})
    return {
        "final_response": warning_message,
        "status": "fallback",
        "messages_history": updated_history,
    }


# ---------------------------------------------------------------------------
# Conditional Edge Routers
# ---------------------------------------------------------------------------


def route_after_info_classifier(state: AgentState) -> str:
    """Branches directly to generator if no retrieval is needed, else routes to source router."""
    if state.needs_retrieval:
        logger.info("Routing -> source_router (external retrieval needed)")
        return "source_router"
    logger.info("Routing -> generator (parametric / conversational path)")
    return "generator"


def route_after_evaluator(state: AgentState) -> str:
    """Evaluates verdict: success, fast-abort on missing info, retry on hallucination, or capped."""
    verdict = state.evaluation

    # 1. Success path
    if not verdict or verdict.is_valid:
        logger.info("Routing -> final_success (verdict valid)")
        return "final_success"

    # 2. Fast-abort path (knowledge does not exist)
    if verdict.failure_type == "missing_info":
        logger.info("Routing -> graceful_fallback (missing_info fast-abort)")
        return "graceful_fallback"

    # 3. Corrective loop (hallucination or unrelated)
    if state.retry_count < state.max_retries:
        logger.info(
            f"Routing -> query_rewriter (corrective loop, retry {state.retry_count}/{state.max_retries})"
        )
        return "query_rewriter"

    # 4. Max retries exceeded
    logger.info("Routing -> capped_fallback (max retries reached)")
    return "capped_fallback"


# ---------------------------------------------------------------------------
# Graph Builder & Compiler
# ---------------------------------------------------------------------------


def build_agrag_graph(
    checkpointer: Optional[Any] = None,
) -> CompiledStateGraph:
    """Builds and compiles the cyclic StateGraph for the AGRAG engine."""
    workflow = StateGraph(AgentState)

    # 1. Register all agent and terminal nodes
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("info_classifier", info_classifier_node)
    workflow.add_node("source_router", source_router_node)
    workflow.add_node("context_merge", context_merge_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("evaluator", response_evaluator_node)
    workflow.add_node("final_success", final_success_node)
    workflow.add_node("graceful_fallback", graceful_fallback_node)
    workflow.add_node("capped_fallback", capped_fallback_node)

    # 2. Add fixed edges
    workflow.add_edge(START, "query_rewriter")
    workflow.add_edge("query_rewriter", "info_classifier")
    workflow.add_edge("source_router", "context_merge")
    workflow.add_edge("context_merge", "generator")
    workflow.add_edge("generator", "evaluator")

    # 3. Add conditional edge from Info Classifier
    workflow.add_conditional_edges(
        "info_classifier",
        route_after_info_classifier,
        {
            "source_router": "source_router",
            "generator": "generator",
        },
    )

    # 4. Add conditional edge from Evaluator (Cyclic feedback loop or terminal)
    workflow.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {
            "final_success": "final_success",
            "graceful_fallback": "graceful_fallback",
            "query_rewriter": "query_rewriter",
            "capped_fallback": "capped_fallback",
        },
    )

    # 5. Terminal edges to END
    workflow.add_edge("final_success", END)
    workflow.add_edge("graceful_fallback", END)
    workflow.add_edge("capped_fallback", END)

    # 6. Compile with checkpointer for multi-turn conversational persistence
    saver = checkpointer if checkpointer is not None else MemorySaver()
    compiled = workflow.compile(checkpointer=saver)
    return compiled


# Module-level default compiled graph
agrag_app = build_agrag_graph()


# ---------------------------------------------------------------------------
# High-Level Pipeline Runner
# ---------------------------------------------------------------------------


async def run_agrag_pipeline(
    user_query: str,
    thread_id: str = "default-thread",
    model_name: str = "deepseek-chat",
    max_retries: int = 2,
    graph: Optional[CompiledStateGraph] = None,
) -> AgentState:
    """Executes the AGRAG state machine end-to-end for a given query."""
    engine = graph or agrag_app
    initial_state = {
        "user_query": user_query,
        "thread_id": thread_id,
        "model_name": model_name,
        "max_retries": max_retries,
    }
    config = {"configurable": {"thread_id": thread_id}}

    result = await engine.ainvoke(initial_state, config=config)
    if isinstance(result, AgentState):
        return result
    return AgentState(**result)
