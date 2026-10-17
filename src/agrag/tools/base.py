"""Base abstract interfaces for retrievers and tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from agrag.state import RetrievedChunk


class BaseRetriever(ABC):
    """Abstract base class for all knowledge retrievers (Vector DB, Web Search, etc.)."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """Retrieve relevant knowledge chunks for a given query."""
        pass


class BaseTool(ABC):
    """Abstract base class for all operational tools and API callers."""

    name: str
    description: str
    parameters_schema: Dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs) -> RetrievedChunk:
        """Execute the tool action and return a structured RetrievedChunk."""
        pass
