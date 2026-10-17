"""Typed State Schemas for AGRAG's LangGraph State Machine."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


SourceType = Literal["vector_db", "web_search", "tool_api"]


class RetrievedChunk(BaseModel):
    """A standardized knowledge chunk retrieved from any source."""

    source_type: SourceType
    source_title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None



class EvaluationVerdict(BaseModel):
    """The diagnostic verdict produced by the response evaluator."""

    is_valid: bool = Field(
        description="True if the response accurately and fully answers the query without hallucination."
    )
    failure_type: Literal["none", "hallucination", "missing_info", "unrelated"] = Field(
        default="none",
        description="The root cause if invalid: hallucination, missing_info, or unrelated.",
    )
    critique: Optional[str] = Field(
        default=None, description="Detailed guidance for the query rewriter if retrying."
    )


class AgentState(BaseModel):
    """Global execution state for the AGRAG cyclic state machine."""

    # Session & Conversation Tracking
    thread_id: str = Field(default="default-thread")
    user_query: str = Field(description="Original user prompt")
    messages_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of past conversation turns: [{'role': 'user'|'assistant', 'content': '...'}]",
    )

    # Node 1 Outputs
    updated_query: Optional[str] = Field(
        default=None, description="Disambiguated and expanded query"
    )

    # Node 2 Outputs
    needs_retrieval: Optional[bool] = Field(
        default=None,
        description="True if query requires external knowledge, False if direct parametric generation suffices",
    )

    # Node 3 & 3.5 Outputs
    selected_sources: List[SourceType] = Field(
        default_factory=list,
        description="Sources selected by source router (e.g. ['vector_db', 'tool_api'])",
    )

    retrieved_chunks: List[RetrievedChunk] = Field(
        default_factory=list, description="All raw chunks collected from active sources"
    )
    fused_context: Optional[str] = Field(
        default=None, description="Provenance-tagged Markdown context block"
    )

    # Node 4 Outputs
    model_name: Literal["deepseek-chat", "deepseek-reasoner"] = Field(
        default="deepseek-chat"
    )
    initial_response: Optional[str] = Field(
        default=None, description="Draft response synthesized by DeepSeek"
    )

    # Node 5 Outputs & Retry Guard
    evaluation: Optional[EvaluationVerdict] = Field(
        default=None, description="Evaluation verdict from evaluator"
    )
    retry_count: int = Field(default=0, description="Number of retry loops executed")
    max_retries: int = Field(default=2, description="Maximum allowed retries")
    critique_history: List[str] = Field(
        default_factory=list, description="Accumulated critiques across retry attempts"
    )

    # Final Output
    final_response: Optional[str] = Field(
        default=None, description="Final user-facing response"
    )
    status: Literal["processing", "completed", "fallback"] = Field(
        default="processing"
    )
