"""Phase 3 Verification Gate Test Suite: LangGraph Assembly & Cyclic Control."""

import pytest
from agrag.graph import build_agrag_graph, run_agrag_pipeline
from langgraph.checkpoint.memory import MemorySaver


class TestPhase3GraphFlow:
    """Automated audit tests for the 4 canonical execution paths and cyclic routing."""

    @pytest.mark.asyncio
    async def test_path1_direct_conversational_no_retrieval(self):
        """Path 1: Conversational greeting should bypass retrieval and complete directly."""
        graph = build_agrag_graph(checkpointer=MemorySaver())
        state = await run_agrag_pipeline(
            user_query="Hello! How are you doing today?",
            thread_id="test-path-1",
            graph=graph,
        )

        assert state.needs_retrieval is False, "Greeting should not trigger retrieval"
        assert state.selected_sources == [], "Direct path should have no selected sources"
        assert state.fused_context is None, "Direct path should have no fused context"
        assert state.status == "completed"
        assert state.final_response is not None
        assert len(state.final_response) > 0
        assert state.retry_count == 0
        # Verify message history updated
        assert len(state.messages_history) == 2
        assert state.messages_history[0]["role"] == "user"
        assert state.messages_history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_path2_clean_single_pass_retrieval(self):
        """Path 2: Valid domain query retrieves context, validates on first try, and completes."""
        graph = build_agrag_graph(checkpointer=MemorySaver())
        state = await run_agrag_pipeline(
            user_query="What is Acme's MFA policy and what is Alex's account status?",
            thread_id="test-path-2",
            graph=graph,
        )

        assert state.needs_retrieval is True, "Domain policy query must trigger retrieval"
        assert len(state.selected_sources) > 0
        assert "vector_db" in state.selected_sources or "tool_api" in state.selected_sources
        assert state.fused_context is not None
        assert state.evaluation is not None
        assert state.evaluation.is_valid is True, "Grounded response must pass evaluation"
        assert state.retry_count == 0, "Clean pass should require 0 retries"
        assert state.status == "completed"
        assert "[Source:" in state.final_response

    @pytest.mark.asyncio
    async def test_path3_missing_info_fast_abort(self):
        """Path 3: Unanswerable query must fast-abort to graceful fallback without looping."""
        graph = build_agrag_graph(checkpointer=MemorySaver())
        state = await run_agrag_pipeline(
            user_query="What is Acme's secret 2029 quantum roadmap?",
            thread_id="test-path-3",
            graph=graph,
        )

        assert state.evaluation is not None
        assert state.evaluation.failure_type == "missing_info"
        assert state.retry_count == 0, "Missing info must NOT waste retry attempts"
        assert state.status == "fallback"
        assert "Notice" in state.final_response or "could not verify" in state.final_response

    @pytest.mark.asyncio
    async def test_path4_hallucination_retry_loop(self):
        """Path 4: Hallucinated response must increment retry count and execute corrective cycle."""
        graph = build_agrag_graph(checkpointer=MemorySaver())
        state = await run_agrag_pipeline(
            user_query="simulate_hallucination query for Acme MFA",
            thread_id="test-path-4",
            max_retries=2,
            graph=graph,
        )

        assert state.evaluation is not None
        # Must have attempted corrective rewrites
        assert state.retry_count >= 1, "Hallucination must increment retry count"
        assert len(state.critique_history) >= 1, "Critique must be recorded in state"
        assert state.status in ["completed", "fallback"]

    @pytest.mark.asyncio
    async def test_path5_multi_turn_conversational_memory(self):
        """Path 5: Thread ID persistence preserves conversation history across turns."""
        graph = build_agrag_graph(checkpointer=MemorySaver())
        shared_thread = "test-session-multi-turn-99"

        # Turn 1
        state_turn1 = await run_agrag_pipeline(
            user_query="Hello, I am Alex from Engineering.",
            thread_id=shared_thread,
            graph=graph,
        )
        assert len(state_turn1.messages_history) == 2

        # Turn 2 using the same thread_id
        state_turn2 = await run_agrag_pipeline(
            user_query="What are my MFA requirements and rotation status?",
            thread_id=shared_thread,
            graph=graph,
        )
        # Turn 2 should carry forward Turn 1 history + Turn 2 turns = 4 turns total
        assert len(state_turn2.messages_history) == 4
        assert "Alex" in state_turn2.messages_history[0]["content"]
        assert state_turn2.status == "completed"

    @pytest.mark.asyncio
    async def test_stream_agrag_pipeline_events(self):
        """Path 6: Verify stream_agrag_pipeline emits expected lifecycle events."""
        from agrag.graph import stream_agrag_pipeline

        events = []
        async for event in stream_agrag_pipeline(
            user_query="What is Acme's MFA policy?",
            thread_id="test-stream-thread",
        ):
            events.append(event)

        assert len(events) > 0
        event_types = [e["event"] for e in events]
        nodes = [e["node"] for e in events]

        # Verify key milestone events are present
        assert "node_complete" in event_types
        assert "routing_decision" in event_types or "info_classifier" in nodes
        assert "final_response" in event_types
        assert "final_success" in nodes or "graceful_fallback" in nodes

    def test_top_level_package_exports(self):
        """Verify top-level clean exports from agrag."""
        import agrag

        assert hasattr(agrag, "run_agrag_pipeline")
        assert hasattr(agrag, "stream_agrag_pipeline")
        assert hasattr(agrag, "build_agrag_graph")
        assert hasattr(agrag, "agrag_app")
        assert hasattr(agrag, "AgentState")

