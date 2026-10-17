# API Specification & Streaming Protocol
## Project: AGRAG (Agentic RAG Engine)
**Version:** 1.0.0  
**Transport:** HTTP/1.1 & HTTP/2 with Server-Sent Events (SSE)  
**Encoding:** UTF-8 / JSON  

---

## 1. Overview

AGRAG exposes a high-performance REST and Server-Sent Events (SSE) API built on FastAPI. The primary endpoint (`/api/chat/stream`) enables client interfaces to render live agent thoughts, node transitions, tool invocations, and streamed answer tokens without polling.

---

## 2. Endpoints

### 2.1 Health Check
- **Path:** `GET /health`
- **Response:**
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "providers": {
      "deepseek": "connected",
      "vector_store": "ready"
    }
  }
  ```

---

### 2.2 Streaming Chat (`/api/chat/stream`)
- **Path:** `POST /api/chat/stream`
- **Headers:**
  - `Content-Type: application/json`
  - `Accept: text/event-stream`

#### Request Payload
```json
{
  "query": "What is our company's MFA policy and what is my current rotation status?",
  "thread_id": "session-usr-10492",
  "model_name": "deepseek-chat",
  "max_retries": 2
}
```

| Field | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `query` | string | Yes | — | The incoming user message. |
| `thread_id` | string | Yes | — | Session identifier for conversational memory checkpointer. |
| `model_name` | string | No | `"deepseek-chat"` | Generator model toggle: `"deepseek-chat"` or `"deepseek-reasoner"`. |
| `max_retries` | integer | No | `2` | Hard cap on corrective retry attempts. |

---

## 3. Server-Sent Events (SSE) Protocol

Events are emitted in real-time as each node in the LangGraph execution tree activates and completes.

### Event Format
```http
event: <event_type>
data: <json_string>

```

### Event Types & Schemas

#### 1. `node_start`
Fires when an agent node begins execution.
```json
{
  "node": "query_rewriter",
  "timestamp": 1773718000.120,
  "status": "running"
}
```

#### 2. `node_complete`
Fires when an agent node completes and updates the state.
```json
{
  "node": "query_rewriter",
  "duration_ms": 780,
  "output": {
    "updated_query": "What is Acme Corp's multi-factor authentication (MFA) rotation policy?"
  }
}
```

#### 3. `routing_decision`
Emitted by Node 2 (`info_need_classifier`) and Node 3 (`source_selector`).
```json
{
  "node": "source_selector",
  "needs_retrieval": true,
  "selected_sources": ["vector_db", "tool_api"]
}
```

#### 4. `retrieval_complete`
Emitted when documents and tool payloads have been retrieved and fused.
```json
{
  "sources_queried": ["vector_db", "tool_api"],
  "chunks_count": 4,
  "sources_summary": [
    {"source": "vector_db", "title": "Security_Policy_2026.pdf", "score": 0.89},
    {"source": "tool_api", "title": "get_user_mfa_status", "score": 1.0}
  ]
}
```

#### 5. `evaluation_verdict`
Emitted by Node 5 (`response_evaluator`).
```json
{
  "is_valid": false,
  "failure_type": "hallucination",
  "critique": "The draft claimed MFA keys expire in 30 days, but Security_Policy_2026 specifies 90 days.",
  "retry_count": 1,
  "action": "retrying_query"
}
```

#### 6. `token` (Streamed Generation)
Emitted token-by-token during synthesis in Node 4.
```json
{
  "delta": "According to the corporate security guidelines, "
}
```

#### 7. `final_response`
Emitted upon completion of the workflow.
```json
{
  "thread_id": "session-usr-10492",
  "response": "According to Acme Corp's Security Policy, all team members must rotate MFA credentials every 90 days. Your current status is active, with your last rotation on August 1st, 2026.",
  "sources_cited": ["Security_Policy_2026.pdf", "get_user_mfa_status"],
  "retries_taken": 1,
  "total_latency_ms": 6820
}
```

#### 8. `error`
Emitted if an unhandled exception or critical API timeout occurs.
```json
{
  "error_code": "DEEPSEEK_RATE_LIMIT",
  "message": "Upstream DeepSeek service returned rate limit error. Retrying with exponential backoff..."
}
```

---

## 4. Error Codes & HTTP Statuses

| Status Code | Error Code | Description |
| :--- | :--- | :--- |
| `400 Bad Request` | `INVALID_INPUT` | Missing `query` or `thread_id`. |
| `422 Unprocessable` | `INVALID_MODEL` | Provided `model_name` is not in allowed list. |
| `502 Bad Gateway` | `UPSTREAM_LLM_FAILURE` | DeepSeek API unreachable or authentication failure. |
| `504 Gateway Timeout`| `RETRIEVAL_TIMEOUT` | Vector DB or external web search timed out beyond 10s. |
