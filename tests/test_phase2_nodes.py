"""Phase 2 Verification Gate Test Suite: Individual Micro-Agent Nodes."""

import pytest
from agrag.nodes.rewriter import query_rewriter_node
from agrag.nodes.info_classifier import info_classifier_node
from agrag.nodes.source_router import source_router_node
from agrag.nodes.context_merge import context_merge_node
from agrag.nodes.generator import generator_node
from agrag.nodes.evaluator import response_evaluator_node
from agrag.state import AgentState, RetrievedChunk


class TestPhase2Nodes:
    """Automated audit tests for Phase 2 individual micro-agent nodes."""

    @pytest.mark.asyncio
    async def test_node1_query_rewriter(self):
        """Verify Node 1 rewrites queries, resolves coreferences, and incorporates critiques."""
        # Case A: Initial ambiguous query
        state_a = AgentState(
            user_query="What is its policy?",
            messages_history=[
                {"role": "user", "content": "I am asking about Multi-Factor Authentication."},
                {"role": "assistant", "content": "Sure, what would you like to know?"},
            ],
        )
        res_a = await query_rewriter_node(state_a)
        assert "updated_query" in res_a
        assert len(res_a["updated_query"]) > 0

        # Case B: Critique integration
        state_b = AgentState(
            user_query="What is MFA rotation?",
            critique_history=["The previous search missed specific rotation deadlines for EU region."],
            retry_count=1,
        )
        res_b = await query_rewriter_node(state_b)
        assert "updated_query" in res_b

    @pytest.mark.asyncio
    async def test_node2_info_classifier(self):
        """Verify Node 2 accurately classifies retrieval need vs parametric shortcuts."""
        # Conversational greeting
        greeting_state = AgentState(user_query="Good morning!", updated_query="Good morning!")
        res_greeting = await info_classifier_node(greeting_state)
        assert res_greeting["needs_retrieval"] is False

        # Domain policy query
        policy_state = AgentState(
            user_query="What is Acme's MFA key rotation requirement?",
            updated_query="What is Acme's MFA key rotation requirement?",
        )
        res_policy = await info_classifier_node(policy_state)
        assert res_policy["needs_retrieval"] is True

    @pytest.mark.asyncio
    async def test_node3_source_router(self):
        """Verify Node 3 correctly selects sources and executes concurrent retrieval."""
        state = AgentState(
            user_query="Check alex account status and mfa requirements",
            updated_query="Check alex account status and mfa requirements",
        )
        res = await source_router_node(state)
        assert "selected_sources" in res
        assert "retrieved_chunks" in res
        assert len(res["retrieved_chunks"]) > 0
        # Check that retrieved items are valid RetrievedChunk instances
        for c in res["retrieved_chunks"]:
            assert isinstance(c, RetrievedChunk)
            assert c.source_type in ["vector_db", "web_search", "tool_api"]

    @pytest.mark.asyncio
    async def test_node3_5_context_merge(self):
        """Verify Node 3.5 formats multi-source chunks into clean provenance-tagged Markdown."""
        chunks = [
            RetrievedChunk(
                source_type="vector_db",
                source_title="Acme Security Policy",
                content="MFA rotation every 90 days required.",
            ),
            RetrievedChunk(
                source_type="tool_api",
                source_title="Employee Record: Alex Mercer",
                content='{"mfa_compliant": true}',
            ),
        ]
        state = AgentState(user_query="test", retrieved_chunks=chunks)
        res = await context_merge_node(state)
        assert "fused_context" in res
        fused = res["fused_context"]
        assert "### [Internal Document: Acme Security Policy]" in fused
        assert "### [Operational Tool API: Employee Record: Alex Mercer]" in fused
        assert "MFA rotation every 90 days required." in fused

    @pytest.mark.asyncio
    async def test_node4_generator(self):
        """Verify Node 4 generates responses with citations and respects model toggle."""
        # Grounded RAG path
        fused = (
            "### [Internal Document: Acme IT Security Policy]\n"
            "All MFA keys must be rotated every 90 calendar days.\n\n"
            "### [Operational Tool API: Employee Record: Alex Mercer (alex)]\n"
            '{"status": "active", "mfa_compliant": true}'
        )
        rag_state = AgentState(
            user_query="What is Alex's status and the MFA policy?",
            fused_context=fused,
            model_name="deepseek-chat",
        )
        res_rag = await generator_node(rag_state)
        assert "initial_response" in res_rag
        assert len(res_rag["initial_response"]) > 20
        assert "[Source:" in res_rag["initial_response"]

        # Direct path (no context)
        direct_state = AgentState(
            user_query="Hello!",
            fused_context=None,
            model_name="deepseek-chat",
        )
        res_direct = await generator_node(direct_state)
        assert "initial_response" in res_direct
        assert len(res_direct["initial_response"]) > 0

    @pytest.mark.asyncio
    async def test_node5_evaluator(self):
        """Verify Node 5 evaluates relevance, detects missing info, and controls retries."""
        # 1. Valid Grounded Response
        fused = "### [Policy]\nMFA rotates every 90 days."
        valid_state = AgentState(
            user_query="What is MFA rotation?",
            initial_response="MFA credentials must be rotated every 90 days. [Source: Policy]",
            fused_context=fused,
            retry_count=0,
        )
        res_valid = await response_evaluator_node(valid_state)
        verdict_valid = res_valid["evaluation"]
        assert verdict_valid.is_valid is True
        assert verdict_valid.failure_type == "none"
        assert res_valid["retry_count"] == 0

        # 2. Missing Information (Fast-Abort Path)
        missing_state = AgentState(
            user_query="What is 2029 quantum roadmap?",
            initial_response="We do not possess roadmap data.",
            fused_context=None,
            retry_count=0,
        )
        res_missing = await response_evaluator_node(missing_state)
        verdict_missing = res_missing["evaluation"]
        assert verdict_missing.is_valid is False
        assert verdict_missing.failure_type == "missing_info"
        # Missing info should NOT increment retry_count
        assert res_missing["retry_count"] == 0

        # 3. Hallucination (Retry Path)
        halluc_state = AgentState(
            user_query="simulate_hallucination query",
            initial_response="MFA rotates once every 10 years.",
            fused_context=fused,
            retry_count=0,
            critique_history=[],
        )
        res_halluc = await response_evaluator_node(halluc_state)
        verdict_halluc = res_halluc["evaluation"]
        assert verdict_halluc.is_valid is False
        assert verdict_halluc.failure_type == "hallucination"
        # Hallucination MUST increment retry_count and record critique
        assert res_halluc["retry_count"] == 1
        assert len(res_halluc["critique_history"]) == 1
