"""Phase 1 Verification Gate Test Suite: Foundation, Environment & Seed Knowledge."""

import pytest
import os
import shutil
from agrag.config import Settings
from agrag.state import AgentState, RetrievedChunk, EvaluationVerdict
from agrag.tools.custom_tools import (
    EmployeeAccountStatusTool,
    IncidentTicketLookupTool,
    tool_registry,
)
from agrag.tools.search import WebSearchRetriever
from agrag.tools.vector_store import ChromaVectorRetriever
from agrag.mock_llm import mock_llm


class TestPhase1Foundation:
    """Automated audit tests for Phase 1 foundational components."""

    def test_config_settings(self):
        """Verify configuration defaults, mock detection, and model toggles."""
        s = Settings(DEEPSEEK_API_KEY=None, MOCK_MODE=False)
        assert s.is_mock is True, "Should detect mock mode when API key is missing"

        s_live = Settings(DEEPSEEK_API_KEY="sk-test-key-12345", MOCK_MODE=False)
        assert s_live.is_mock is False, "Should recognize live mode when API key is present"
        assert s_live.deepseek_default_model == "deepseek-chat"
        assert s_live.max_retries == 2

    def test_agent_state_schema(self):
        """Verify strongly-typed Pydantic contracts for AgentState and verdicts."""
        chunk = RetrievedChunk(
            source_type="vector_db",
            source_title="Acme Security Policy",
            content="MFA rotation every 90 days",
            score=0.95,
        )
        assert chunk.source_type == "vector_db"
        assert chunk.score == 0.95

        verdict = EvaluationVerdict(
            is_valid=False,
            failure_type="hallucination",
            critique="Invented unverified claim.",
        )
        assert verdict.is_valid is False
        assert verdict.failure_type == "hallucination"

        state = AgentState(
            user_query="What is my MFA status?",
            retrieved_chunks=[chunk],
            evaluation=verdict,
        )
        assert state.user_query == "What is my MFA status?"
        assert len(state.retrieved_chunks) == 1
        assert state.retry_count == 0
        assert state.status == "processing"

    @pytest.mark.asyncio
    async def test_custom_tools_execution(self):
        """Verify Acme Cloud Corp operational tools (employee status & incident search)."""
        emp_tool = EmployeeAccountStatusTool()

        # 1. Test existing compliant employee
        alex_res = await emp_tool.execute(username="alex")
        assert alex_res.source_type == "tool_api"
        assert "Alex Mercer" in alex_res.content
        assert "mfa_compliant\": true" in alex_res.content.lower()

        # 2. Test non-compliant employee
        sarah_res = await emp_tool.execute(username="sarah")
        assert "mfa_compliant\": false" in sarah_res.content.lower()

        # 3. Test non-existent employee
        unknown_res = await emp_tool.execute(username="unknown_user")
        assert "not found" in unknown_res.content.lower()

        # 4. Test incident ticket lookup
        inc_tool = IncidentTicketLookupTool()
        ticket_res = await inc_tool.execute(query="vpn")
        assert ticket_res.source_type == "tool_api"
        assert "IT-10294" in ticket_res.content

        # 5. Test ToolRegistry description prompt
        prompt_desc = tool_registry.get_tools_description_prompt()
        assert "get_employee_account_status" in prompt_desc
        assert "lookup_incident_tickets" in prompt_desc

    @pytest.mark.asyncio
    async def test_web_search_retriever(self):
        """Verify WebSearchRetriever returns standardized RetrievedChunk objects."""
        searcher = WebSearchRetriever()
        chunks = await searcher.retrieve("MFA security guidelines", top_k=2)
        assert len(chunks) > 0
        assert chunks[0].source_type == "web_search"
        assert len(chunks[0].content) > 10

    @pytest.mark.asyncio
    async def test_vector_store_retrieval(self):
        """Verify ChromaDB vector store indexing and semantic recall on Acme seed docs."""
        test_dir = "./data/test_chroma_phase1"
        try:
            retriever = ChromaVectorRetriever(persist_dir=test_dir)
            chunks = await retriever.retrieve("MFA key rotation policy", top_k=2)
            assert len(chunks) > 0
            assert chunks[0].source_type == "vector_db"
            # Verify recall of 90-day policy from seed docs
            combined_text = " ".join(c.content for c in chunks)
            assert "90 calendar days" in combined_text or "Multi-Factor Authentication" in combined_text
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir, ignore_errors=True)

    def test_mock_llm_engine(self):
        """Verify MockLLMEngine deterministic generation, classification, and evaluation."""
        # 1. Info need classification
        assert mock_llm.classify_info_need("Hello there!") is False
        assert mock_llm.classify_info_need("What is 2+2?") is False
        assert mock_llm.classify_info_need("What is our MFA policy?") is True

        # 2. Source routing
        sources = mock_llm.route_sources("Check alex account status and mfa policy")
        assert "tool_api" in sources
        assert "vector_db" in sources

        # 3. Grounded generation
        fused = "### [Tool: Employee Record]\nAlex Mercer mfa_compliant: true\n### [Policy]\nRotate every 90 days."
        resp = mock_llm.generate_response("Alex MFA", fused)
        assert "Alex Mercer" in resp
        assert "[Source:" in resp

        # 4. Evaluation logic
        # Clean response
        clean_eval = mock_llm.evaluate_response("What is MFA policy?", resp, fused)
        assert clean_eval.is_valid is True
        assert clean_eval.failure_type == "none"

        # Missing info detection
        missing_eval = mock_llm.evaluate_response("What is 2029 quantum roadmap?", "Some draft", None)
        assert missing_eval.is_valid is False
        assert missing_eval.failure_type == "missing_info"

        # Hallucination simulation
        halluc_eval = mock_llm.evaluate_response("simulate_hallucination query", "Draft", fused)
        assert halluc_eval.is_valid is False
        assert halluc_eval.failure_type == "hallucination"
