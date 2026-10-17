# Technical Architecture & System Specification
## Project: AGRAG (Agentic RAG Engine)
**Version:** 1.0.0  
**Framework:** LangGraph (Python 3.11+)  
**Primary LLM Engine:** DeepSeek API (`api.deepseek.com`)  

---

## 1. System Architecture Overview

AGRAG implements an event-driven, cyclic directed acyclic graph (with controlled loops) using **LangGraph**. The state machine encapsulates conversational history, intermediate query rewrites, multi-source retrieved context, draft answers, evaluation verdicts, and retry diagnostics.

```
                           +----------------------+
                           |   AgentState Input   |
                           +-----------+----------+
                                       |
                                       v
                           +----------------------+
                           | Node 1: QueryWriter  |<-----------------+
                           +-----------+----------+                  |
                                       |                             |
                                       v                             |
                           +----------------------+                  |
                           | Node 2: InfoCheck    |                  |
                           +-----+------------+---+                  |
                              No |            | Yes                  |
                                 |            v                      |
                                 | +----------------------+          |
                                 | | Node 3: SourceRouter |          |
                                 | +----------+-----------+          |
                                 |            |                      |
                                 |            +----------+           |
                                 |            |          |           |
                                 |            v          v           |
                                 |         [Chroma]   [WebSearch]    |
                                 |            |          |           |
                                 |            +----+-----+           |
                                 |                 |                 |
                                 |                 v                 |
                                 |     +-----------------------+     |
                                 |     | Node 3.5: ContextMerge|     |
                                 |     +-----------+-----------+     |
                                 |                 |                 |
                                 v                 v                 |
                           +------------------------------+          |
                           |   Node 4: DeepSeekGenerator  |          |
                           +--------------+---------------+          |
                                          |                          |
                                          v                          |
                           +------------------------------+          |
                           |       Node 5: Evaluator      |          |
                           +--------------+---------------+          |
                                          |                          |
               +--------------------------+-----------------------+  |
               | is_valid == True         | failure == missing_info|  | (hallucination
               v                          v                       v  |  and retry < 2)
     +-------------------+      +-------------------+      +---------+---------+
     |   Final Response  |      |   Fast Fallback   |      |  Increment Retry  |
     |      (Success)    |      | (Knowledge Absent)|      | & Return Critique |
     +-------------------+      +-------------------+      +-------------------+
```

---

## 2. State Schema Contract (`AgentState`)

The graph operates on an immutable, strongly-typed Pydantic state model passed across all nodes.

```python
from typing import Annotated, List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import operator

class RetrievedChunk(BaseModel):
    source_type: Literal["vector_db", "web_search", "tool_api"]
    source_title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

class EvaluationVerdict(BaseModel):
    is_valid: bool = Field(description="True if the response accurately and fully answers the query without hallucination.")
    failure_type: Literal["none", "hallucination", "missing_info", "unrelated"] = Field(
        default="none",
        description="The root cause if invalid: hallucination (claims unsupported by context), missing_info (answer not present in sources), unrelated (answers the wrong question)."
    )
    critique: Optional[str] = Field(default=None, description="Detailed guidance for the rewriter if retrying.")

class AgentState(BaseModel):
    # Session & Conversation Tracking
    thread_id: str
    user_query: str
    messages_history: List[Dict[str, str]] = Field(default_factory=list)
    
    # Node 1 Outputs
    updated_query: Optional[str] = None
    
    # Node 2 Outputs
    needs_retrieval: Optional[bool] = None
    
    # Node 3 & 3.5 Outputs
    selected_sources: List[str] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    fused_context: Optional[str] = None
    
    # Node 4 Outputs
    model_name: Literal["deepseek-chat", "deepseek-reasoner"] = "deepseek-chat"
    initial_response: Optional[str] = None
    
    # Node 5 Outputs & Retry Guard
    evaluation: Optional[EvaluationVerdict] = None
    retry_count: int = 0
    max_retries: int = 2
    critique_history: List[str] = Field(default_factory=list)
    
    # Final Output
    final_response: Optional[str] = None
    status: Literal["processing", "completed", "fallback"] = "processing"
```

---

## 3. Node Specifications & Contracts

### Node 1: `query_rewriter`
- **Model:** `deepseek-chat` (DeepSeek-V3)
- **Role:** Analyzes `messages_history` + `user_query` + latest `critique` (if in a retry loop).
- **Behavior:**
  - If `retry_count == 0`: Resolves ambiguous references (e.g., "what about its price?" $\rightarrow$ "What is the price of Acme Cloud Tier 2?").
  - If `retry_count > 0`: Incorporates the evaluator's critique to broaden or refocus the search query without losing original user intent.
- **Output:** Updates `updated_query`.

### Node 2: `info_need_classifier`
- **Model:** `deepseek-chat` (DeepSeek-V3) structured output.
- **Prompt Objective:** Determines if `updated_query` requires external empirical knowledge.
- **Criteria:**
  - `needs_retrieval = False`: Conversational greetings, standard mathematical operations, pure logic problems, code syntax explanation.
  - `needs_retrieval = True`: Company policies, product catalogs, live data, external facts, domain-specific questions.
- **Output:** Updates `needs_retrieval`.

### Node 3: `source_selector`
- **Model:** `deepseek-chat` (DeepSeek-V3) structured routing.
- **Sources Available:**
  1. `vector_db`: Internal company documentation, FAQs, product knowledge base (via ChromaDB).
  2. `web_search`: Up-to-date public information, breaking news, external documentation (via DuckDuckGo / Tavily).
  3. `tool_api`: Structured transactional data (e.g., order status, internal metrics).
- **Execution:** Dispatches concurrent asynchronous tasks (`asyncio.gather`) to selected sources.
- **Output:** Populates `retrieved_chunks`.

### Node 3.5: `context_fusion`
- **Logic:** Pure Python deterministic transformation.
- **Format:** Merges all `retrieved_chunks` into a clean Markdown block with explicit provenance tagging:
  ```markdown
  ### [Internal Document: Security_Policy_v2.pdf]
  Employees must rotate MFA keys every 90 days...

  ### [Live Web Search: https://aws.amazon.com/security]
  AWS IAM announces new automated key rotation protocols...

  ### [Tool API: get_user_status]
  {"user": "alex", "mfa_last_rotated": "2026-08-01", "status": "active"}
  ```
- **Output:** Populates `fused_context`.

### Node 4: `deepseek_generator`
- **Model:** `deepseek-chat` or `deepseek-reasoner` (based on state toggle).
- **Prompt Contract:**
  - Instructs model to ground its response **strictly** in `fused_context` if present.
  - If context is empty or absent (`needs_retrieval == False`), relies on parametric reasoning.
  - Enforces explicit citations: `[Source: <Title>]`.
- **Output:** Populates `initial_response`.

### Node 5: `response_evaluator`
- **Model:** `deepseek-chat` (DeepSeek-V3) with JSON mode.
- **Evaluation Dimensions:**
  1. **Answer Relevance:** Does the answer directly address `updated_query`?
  2. **Groundedness / Hallucination:** Are all stated facts and claims verified by `fused_context`?
  3. **Information Completeness:** Did the source actually have the answer, or did the model attempt to invent details?
- **Output:** Populates `evaluation` with `EvaluationVerdict`.

---

## 4. Conditional Edge & Routing Logic

```python
def route_after_evaluation(state: AgentState) -> str:
    verdict = state.evaluation
    
    # 1. Success path
    if verdict and verdict.is_valid:
        return "generate_final_success"
        
    # 2. Fast-abort: Source does not contain the answer
    if verdict and verdict.failure_type == "missing_info":
        return "generate_graceful_fallback"
        
    # 3. Retry loop for hallucination or unrelated answer
    if state.retry_count < state.max_retries:
        return "query_rewriter"
        
    # 4. Hard cap reached
    return "generate_capped_fallback"
```

---

## 5. Project Directory Structure

```
d:/projects/agy/agrag/
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   └── API_SPEC.md
├── src/
│   └── agrag/
│       ├── __init__.py
│       ├── config.py                 # Pydantic Settings & API keys
│       ├── state.py                  # AgentState & Pydantic models
│       ├── graph.py                  # LangGraph StateGraph builder
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── rewriter.py           # Node 1
│       │   ├── info_classifier.py    # Node 2
│       │   ├── source_router.py      # Node 3
│       │   ├── context_merge.py      # Node 3.5
│       │   ├── generator.py          # Node 4
│       │   └── evaluator.py          # Node 5
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py               # BaseRetriever & BaseTool
│       │   ├── vector_store.py       # ChromaDB implementation
│       │   ├── search.py             # DuckDuckGo / Tavily implementation
│       │   └── custom_tools.py       # Sample Python tool functions
│       └── server/
│           ├── __init__.py
│           ├── app.py                # FastAPI web service
│           └── sse.py                # Event generator for streaming
├── cli/
│   └── main.py                       # Rich-based interactive terminal app
├── tests/
│   ├── test_rewriter.py
│   ├── test_evaluator.py
│   └── test_graph_flow.py
├── pyproject.toml
├── .env.example
└── README.md
```

## 6. Seed Domain, Embeddings & Offline Mode


### 6.1 Vector Embeddings
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Execution:** 100% local in-memory/on-device execution via HuggingFace/Chroma. Zero external API cost or token overhead.

### 6.2 Default Seed Knowledge Domain (Acme Cloud Corp)
- **Vector DB Documents:**
  - `it_security_policy.md`: MFA key rotation (90 days), session timeouts (15 minutes), password complexity requirements.
  - `remote_work_handbook.md`: VPN guidelines, approved cloud storage endpoints, travel laptop security protocols.
- **Tools/APIs:**
  - `get_employee_account_status(username: str)`: Returns real-time user records (MFA status, last rotation timestamp, department, active flags).
  - `lookup_incident_tickets(query: str)`: Returns mock active Jira/ServiceNow tickets.
- **Web Search Integration:**
  - DuckDuckGo / Tavily live search with fallback to mock search if internet connectivity is unavailable.

### 6.3 Offline / Mock Engine Mode (`--mock`)
- When `DEEPSEEK_API_KEY` is omitted or `--mock` flag is set, the engine routes through deterministic mock LLM generators.
- Allows complete verification of the LangGraph state machine, SSE streaming events, retry loops, and the CLI interface without requiring API credentials.

