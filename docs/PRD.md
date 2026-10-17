# Product Requirements Document (PRD)
## Project: AGRAG (Agentic RAG Engine)
**Status:** Approved Specification  
**Version:** 1.0.0  
**Target Engine:** LangGraph (Python) + DeepSeek API  

---

## 1. Executive Summary & Objective

**AGRAG** is an enterprise-grade **Agentic Retrieval-Augmented Generation (Agentic RAG)** engine that moves beyond static, one-shot vector search. By orchestrating a cyclic, multi-agent state machine, AGRAG dynamically:
1. Rewrites and disambiguates incoming queries based on multi-turn conversation history.
2. Decides whether external retrieval is genuinely needed or if parametric reasoning suffices.
3. Selectively routes retrieval across three heterogeneous knowledge silos: **Vector Database**, **Live Internet Search**, and **Custom Tools/APIs**.
4. Synthesizes answers using DeepSeek models (`deepseek-chat` or `deepseek-reasoner`).
5. Self-evaluates output relevance and groundedness in real-time, executing targeted query rewrites upon hallucinations or fast-aborting when information is verified as non-existent.

---

## 2. Problem Statement & User Personas

### 2.1 The Problem with Naive RAG
- **Static Retrieval Waste:** Blindly querying vector databases for conversational greetings, basic calculations, or pure reasoning queries adds latency, cost, and context noise.
- **Single-Source Myopia:** Real enterprise intelligence requires combining static internal documentation (Vector DB), real-time external facts (Web Search), and transactional live state (APIs).
- **Unchecked Hallucination:** Standard RAG cannot verify whether the generator actually adhered to the retrieved context or hallucinated plausible-sounding falsehoods.
- **Deadly Retry Loops:** Without dynamic failure classification, naive corrective RAG loops endlessly when a user asks for information not present in any source.

### 2.2 Target Personas
1. **Enterprise Knowledge Worker:** Needs accurate answers from company handbooks, live ERP statuses, and industry regulations without hallucinated facts.
2. **Technical Copilot / Researcher:** Requires deep synthesis across documentation, live web articles, and code repositories with transparent step-by-step reasoning.
3. **Application Developer:** Demands a modular, plug-and-play Python library with both an interactive CLI for testing and an SSE-compliant FastAPI server for frontend integration.

---

## 3. Product Goals & Success Metrics

### 3.1 Quantitative Metrics
- **First-Token Latency (Direct Path):** $< 2.5$ seconds P50 when no retrieval is needed.
- **End-to-End Latency (Single-Retrieval Path):** $< 8.0$ seconds P50 using `deepseek-chat`.
- **Hallucination Rate:** $< 2\%$ across benchmark evaluation queries with authoritative context.
- **Infinite Loop Prevention:** Exactly $0\%$ infinite loops. Guaranteed termination within $\le 2$ retry attempts or immediate short-circuit on verified information absence.

### 3.2 Qualitative Metrics
- **Transparency:** The user or frontend client must receive continuous real-time progress indicators via Server-Sent Events (SSE) representing every agent thought and tool call.
- **Modularity:** Developers must be able to switch Vector DBs (Chroma vs Qdrant) or search providers (DuckDuckGo vs Tavily) without altering the core graph logic.

---

## 4. System Flow & Feature Requirements

### 4.1 Feature Breakdown

```
[User Query] 
     │
     ▼
[Node 1: History-Aware Query Rewriter]
     │
     ▼
[Node 2: Retrieval Need Classifier]
    ├── (No)  ──► [Node 4: DeepSeek Direct Generation]
    └── (Yes) ──► [Node 3: Source Selector Agent]
                        ├── Vector DB (ChromaDB)
                        ├── Internet Search (DuckDuckGo / Tavily)
                        └── Tools & APIs (Python Function Registry)
                        └──► [Context Fusion (Tagged Markdown)]
                        └──► [Node 4: DeepSeek RAG Generation]
                                       │
                                       ▼
                         [Node 5: Groundedness & Relevance Evaluator]
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
           [Passed]             [Missing Info]          [Hallucination]
                │                      │                      │
         (Final Response)       (Fast Fallback)         (Retry $\le$ 2)
                                                              │
                                                              ▼
                                                        (Back to Node 1)
```

### 4.2 Functional Requirements

| ID | Requirement | Priority |
| :--- | :--- | :--- |
| **FR-01** | **Multi-Turn Coreference Resolution:** Node 1 must resolve ambiguous pronouns and references using conversation history before any routing or search. | P0 |
| **FR-02** | **Conditional Retrieval Bypass:** Node 2 must skip retrieval if the query is a greeting, basic math, or pure logical deduction. | P0 |
| **FR-03** | **Parallel Multi-Source Retrieval:** Node 3 must support concurrent retrieval across Vector DB, Web, and Tools without blocking the event loop. | P0 |
| **FR-04** | **Provenance-Tagged Context:** Merged context must clearly identify the originating source of each snippet to prevent context poisoning. | P0 |
| **FR-05** | **Configurable DeepSeek Models:** The generator must support runtime switching between `deepseek-chat` (V3) and `deepseek-reasoner` (R1). | P1 |
| **FR-06** | **Granular Diagnostic Evaluation:** Node 5 must classify evaluation failures into `none`, `hallucination`, `missing_info`, or `unrelated`. | P0 |
| **FR-07** | **Smart Circuit Breaker:** If evaluation indicates `missing_info`, the engine must abort immediately to a graceful gap explanation rather than re-querying. | P0 |
| **FR-08** | **Real-Time Step Streaming:** The engine must stream intermediate node steps (`query_rewritten`, `retrieving_sources`, `evaluating`) via SSE. | P0 |
| **FR-09** | **Interactive Developer CLI:** A local terminal CLI utilizing `rich` to visualize the execution tree, latency per node, and streaming responses. | P1 |

---

## 5. Non-Functional Requirements

1. **Security & Secrets:** API keys (`DEEPSEEK_API_KEY`, search keys) must be loaded strictly from `.env` or system environment variables; never hardcoded.
2. **Resilience & Graceful Degradation:** If internet search or a specific tool fails or times out, the engine must continue with available local vector context rather than crashing.
3. **Portability:** Written in Python $\ge 3.11$ managed via `uv` for reproducible virtual environments.
4. **Clean Abstractions:** Standardized `BaseRetriever` and `BaseTool` interfaces to guarantee cloud-readiness.

---

## 6. Assumptions & Out-of-Scope (M0 Release)

### In-Scope (M0)
- Python LangGraph StateGraph engine.
- DeepSeek API integration (OpenAI client format).
- Local ChromaDB vector storage with sample enterprise seed data.
- DuckDuckGo / Tavily web search integration.
- Mock operational tools (e.g. system status / order lookup).
- FastAPI backend with SSE streaming endpoints.
- Rich-based interactive terminal runner.

### Out-of-Scope (Future Milestones)
- Multi-user authentication & RBAC.
- Complex multimodal retrieval (images/video).
- Distributed document parsing pipelines (OCR, Unstructured).
