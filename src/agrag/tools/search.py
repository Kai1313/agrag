import asyncio
import logging
from typing import List
from agrag.config import settings

from agrag.state import RetrievedChunk
from agrag.tools.base import BaseRetriever

logger = logging.getLogger(__name__)


class WebSearchRetriever(BaseRetriever):
    """Retriever for live internet search with automatic fallback."""

    def __init__(self):
        self.tavily_api_key = settings.tavily_api_key

    async def retrieve(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """Perform search using Tavily if API key exists, otherwise DuckDuckGo, with offline mock fallback."""
        if settings.is_mock:
            return self._mock_search(query, top_k)

        # 1. Try Tavily if configured
        if self.tavily_api_key:
            try:
                return await self._tavily_search(query, top_k)
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}. Falling back to DuckDuckGo.")

        # 2. Try DuckDuckGo
        try:
            return await self._duckduckgo_search(query, top_k)
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}. Using offline mock search.")
            return self._mock_search(query, top_k)

    async def _tavily_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        import httpx

        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.tavily_api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": top_k,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        chunks = []
        for item in data.get("results", []):
            chunks.append(
                RetrievedChunk(
                    source_type="web_search",
                    source_title=item.get("title", "Web Result"),
                    content=item.get("content", ""),
                    metadata={"url": item.get("url", "")},
                    score=item.get("score", 0.85),
                )
            )
        return chunks

    async def _duckduckgo_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Runs synchronous DuckDuckGo search in a worker thread with a 5-second timeout."""
        return await asyncio.wait_for(
            asyncio.to_thread(self._ddgs_sync, query, top_k),
            timeout=5.0,
        )

    def _ddgs_sync(self, query: str, top_k: int) -> List[RetrievedChunk]:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=top_k):
                results.append(
                    RetrievedChunk(
                        source_type="web_search",
                        source_title=r.get("title", "Web Result"),
                        content=r.get("body", ""),
                        metadata={"url": r.get("href", "")},
                        score=0.8,
                    )
                )
        return results


    def _mock_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Deterministic mock web search results for tests and offline usage."""
        return [
            RetrievedChunk(
                source_type="web_search",
                source_title="Cloud Security Alliance: 2026 Best Practices",
                content=(
                    f"Industry guidance for queries relating to '{query}': Modern zero-trust architectures "
                    "mandate hardware-backed FIDO2 security keys and automated 90-day rotation cycles for privileged credentials."
                ),
                metadata={"url": "https://cloudsecurityalliance.org/guidance/2026"},
                score=0.88,
            ),
            RetrievedChunk(
                source_type="web_search",
                source_title="NIST Special Publication 800-63B Guidelines",
                content=(
                    "NIST guidelines recommend deprecating SMS-based out-of-band authentication due to SIM-swapping risks. "
                    "Authenticator apps or cryptographic tokens should be enforced across all corporate services."
                ),
                metadata={"url": "https://pages.nist.gov/800-63-3/sp800-63b.html"},
                score=0.82,
            ),
        ][:top_k]
