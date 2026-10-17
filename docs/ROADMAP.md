# Implementation Roadmap & Audit Plan
## Project: AGRAG (Agentic RAG Engine)
**Version:** 1.0.0  
**Status:** In Progress — Phase-by-Phase Audited Execution  

---

## Overview & Audit Protocol

To ensure 100% architectural fidelity and zero regressions, the implementation is divided into **4 distinct, auditable phases**. 

### The Verification Gate Protocol:
1. **Build Step:** Files are created and unit/integration code written.
2. **Automated Test Gate:** A dedicated test suite (`pytest tests/test_phaseX.py`) is run and must pass 100% of assertions.
3. **Audit Checkpoint Report:** A structured report is presented to the user showing:
   - Files created / modified.
   - Test execution results & assertion metrics.
   - Live inspection snippet / output demo.
4. **Human Sign-off:** The agent explicitly waits for user approval before moving to the next phase.

---

## Phase Checklist & Milestones

### [x] Phase 1: Foundation, Environment & Seed Knowledge
- [x] Initialize Python environment with `uv` and configure `pyproject.toml` (LangGraph, FastAPI, ChromaDB, Rich, Pytest).
- [x] Implement `src/agrag/config.py` (Pydantic Settings, environment variables, mock mode flags).
- [x] Implement `src/agrag/state.py` (Typed Pydantic `AgentState`, `RetrievedChunk`, `EvaluationVerdict`).
- [x] Implement `src/agrag/tools/base.py` (`BaseRetriever`, `BaseTool` interfaces).
- [x] Implement `src/agrag/tools/vector_store.py` (ChromaDB + Acme Cloud Corp seed data).
- [x] Implement `src/agrag/tools/search.py` (DuckDuckGo / Tavily with fallback).
- [x] Implement `src/agrag/tools/custom_tools.py` (`get_employee_account_status`, `lookup_incident_tickets`).
- [x] Implement mock LLM response generator for zero-credential offline testing.
- [x] **Verification Gate 1:** `tests/test_phase1_foundation.py` passed 100% of assertions (6/6 tests).

---

### [ ] Phase 2: Individual Micro-Agent Nodes
- [ ] Implement Node 1: `src/agrag/nodes/rewriter.py` (Query expansion, coreference resolution, critique integration).
- [ ] Implement Node 2: `src/agrag/nodes/info_classifier.py` (Retrieval need classification: yes/no).
- [ ] Implement Node 3: `src/agrag/nodes/source_router.py` (Dynamic source routing & parallel fan-out).
- [ ] Implement Node 3.5: `src/agrag/nodes/context_merge.py` (Provenance-tagged Markdown context fusion).
- [ ] Implement Node 4: `src/agrag/nodes/generator.py` (DeepSeek-chat / reasoner generation with source grounding).
- [ ] Implement Node 5: `src/agrag/nodes/evaluator.py` (Relevance & hallucination evaluation, typed failure classification).
- [ ] **Verification Gate 2:** `tests/test_phase2_nodes.py` validating each node's input/output contracts, parsing reliability, and state updates.

---

### [ ] Phase 3: LangGraph Assembly & Cyclic Control
- [ ] Implement `src/agrag/graph.py` (Compiling LangGraph `StateGraph` with state reducer).
- [ ] Implement conditional routing edges:
  - Routing after `info_classifier` (Direct path vs Retrieval path).
  - Routing after `evaluator`:
    - `is_valid == True` $\rightarrow$ Final success.
    - `failure_type == "missing_info"` $\rightarrow$ Immediate graceful gap fallback.
    - `failure_type in ["hallucination", "unrelated"]` and `retry_count < max_retries` $\rightarrow$ Loop back to `rewriter`.
    - `retry_count >= max_retries` $\rightarrow$ Max-retries capped fallback.
- [ ] Integrate session memory checkpointer (`MemorySaver` / SQLite) with `thread_id`.
- [ ] **Verification Gate 3:** `tests/test_phase3_graph.py` asserting the 4 canonical execution paths:
  1. Direct conversational path (greeting/math $\rightarrow$ no RAG).
  2. Clean single-pass retrieval path.
  3. Missing info fast-abort path.
  4. Hallucination retry loop (critique injected $\rightarrow$ rewrite $\rightarrow$ resolution or capped).

---

### [ ] Phase 4: Serving & Streaming UX
- [ ] Implement `src/agrag/server/app.py` (FastAPI REST service with health & chat endpoints).
- [ ] Implement `src/agrag/server/sse.py` (Event generator streaming node starts, routing choices, tool updates, tokens, and final answer).
- [ ] Implement `cli/main.py` (Rich-based interactive terminal client with live spinners, step trees, and token streaming).
- [ ] Provide `.env.example` and end-to-end integration tests.
- [ ] **Verification Gate 4:** `tests/test_phase4_e2e.py` validating the FastAPI SSE stream and CLI execution.

---


## Audit Logs

### Phase 1 Audit Checkpoint — [PASSED 6/6]
- **Date & Timestamp:** 2026-10-17T08:44:00Z
- **Test Suite:** `tests/test_phase1_foundation.py`
- **Results:**
  - `test_config_settings`: PASSED
  - `test_agent_state_schema`: PASSED
  - `test_custom_tools_execution`: PASSED
  - `test_web_search_retriever`: PASSED
  - `test_vector_store_retrieval`: PASSED
  - `test_mock_llm_engine`: PASSED
- **Artifacts Verified:**
  - Pydantic models validate typed data correctly.
  - ChromaDB accurately indexed and semantically retrieved 90-day MFA requirements from Acme seed documentation.
  - Acme operational tools (`alex`, `sarah`, `lookup_incident_tickets`) return structured JSON payloads with compliance flags.
  - Mock engine generates grounded responses with source citations and diagnostic evaluation verdicts.

