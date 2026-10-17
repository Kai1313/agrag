# AGRAG (Agentic RAG Engine)

An enterprise-grade Agentic Retrieval-Augmented Generation (Agentic RAG) engine built with LangGraph, DeepSeek API, and local ChromaDB.

## Features
- **Dynamic Adaptive RAG**: Routes queries based on info necessity and source suitability.
- **Heterogeneous Retrieval**: Multi-source querying across Vector DB, Live Web, and Tools/APIs.
- **Real-Time Step Streaming**: Server-Sent Events (SSE) emitting intermediate thoughts and node states.
- **Corrective Self-Evaluation**: Identifies hallucinations and missing info with dynamic circuit breakers.
- **Dual Delivery**: Interactive Rich terminal CLI and FastAPI streaming REST API.

## Documentation
- [PRD.md](docs/PRD.md)
- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [API_SPEC.md](docs/API_SPEC.md)
- [ROADMAP.md](docs/ROADMAP.md)
- [flow.md](docs/flow.md)

