"""ChromaDB Vector Store Retriever with Acme Cloud Corp seed data."""

import logging
import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from agrag.config import settings
from agrag.state import RetrievedChunk
from agrag.tools.base import BaseRetriever

logger = logging.getLogger(__name__)

# Acme Cloud Corp Seed Documents
ACME_SEED_DOCUMENTS = [
    {
        "id": "doc_mfa_policy_01",
        "title": "Acme IT Security Policy - MFA Requirements (Sec. 4.2)",
        "content": (
            "All Acme Cloud Corp employees, contractors, and third parties accessing corporate infrastructure "
            "are required to enforce Multi-Factor Authentication (MFA). MFA credentials must be rotated at least "
            "every 90 calendar days. Approved authentication methods include FIDO2 hardware security keys (e.g., YubiKey) "
            "and approved TOTP Authenticator apps (Google Authenticator, 1Password). SMS and voice call verification "
            "are strictly prohibited due to SIM-swapping vulnerabilities."
        ),
        "category": "security_policy",
    },
    {
        "id": "doc_session_policy_02",
        "title": "Acme IT Security Policy - Session Timeout & Inactivity (Sec. 4.5)",
        "content": (
            "All authenticated sessions on internal portals, cloud consoles (AWS/GCP), and corporate laptops "
            "must automatically lock after 15 minutes of user inactivity. Screen savers must require biometric "
            "or password re-authentication. Password complexity mandates a minimum of 16 characters with at least "
            "three character categories (uppercase, lowercase, numbers, symbols)."
        ),
        "category": "security_policy",
    },
    {
        "id": "doc_remote_work_01",
        "title": "Acme Remote Work & Travel Security Handbook (Sec. 2.1)",
        "content": (
            "Remote employees must always connect to the corporate network via Acme WireGuard VPN prior to accessing "
            "internal GitLab, Jira, or production environments. Split-tunneling is disabled by policy. "
            "Employees traveling internationally must register their travel with SecOps at least 5 business days "
            "in advance to receive a loaner travel laptop and temporary geo-fenced travel access."
        ),
        "category": "remote_work",
    },
    {
        "id": "doc_cloud_storage_01",
        "title": "Acme Data Governance & Storage Policy (Sec. 6.3)",
        "content": (
            "Customer data and proprietary code must never be uploaded to unapproved third-party cloud storage "
            "(including personal Google Drive, Dropbox, or public Pastebin). Approved cloud storage consists solely "
            "of Acme's enterprise Google Drive and AWS S3 buckets with server-side KMS encryption (SSE-KMS) enabled."
        ),
        "category": "data_governance",
    },
]


class ChromaVectorRetriever(BaseRetriever):
    """Vector database retriever powered by ChromaDB and local sentence-transformers."""

    COLLECTION_NAME = "acme_knowledge_base"

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._init_collection()

    def _init_collection(self):
        """Initializes the collection and seeds Acme Cloud Corp documents if empty."""
        try:
            from chromadb.utils import embedding_functions

            self.embedding_fn = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=settings.embedding_model_name
                )
            )
        except Exception as e:
            logger.warning(
                f"Could not load SentenceTransformerEmbeddingFunction ({e}). Using default Chroma embedding."
            )
            self.embedding_fn = None

        if self.embedding_fn:
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
            )
        else:
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME
            )

        # Seed data if collection is empty
        if self.collection.count() == 0:
            logger.info("Seeding Acme Cloud Corp knowledge base documents into ChromaDB...")
            self.collection.add(
                ids=[doc["id"] for doc in ACME_SEED_DOCUMENTS],
                documents=[doc["content"] for doc in ACME_SEED_DOCUMENTS],
                metadatas=[
                    {"title": doc["title"], "category": doc["category"]}
                    for doc in ACME_SEED_DOCUMENTS
                ],
            )

    async def retrieve(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """Query ChromaDB and return structured RetrievedChunk objects."""
        # Check if in mock mode and collection empty
        if settings.is_mock and self.collection.count() == 0:
            return self._mock_retrieve(query, top_k)

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, max(1, self.collection.count())),
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}. Falling back to mock retrieve.")
            return self._mock_retrieve(query, top_k)

        chunks: List[RetrievedChunk] = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for i, doc_text in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                title = meta.get("title", f"Document {i+1}")
                # Convert distance to approximate similarity score
                dist = distances[i] if i < len(distances) else 0.5
                score = round(max(0.0, 1.0 - (dist / 2.0)), 3)

                chunks.append(
                    RetrievedChunk(
                        source_type="vector_db",
                        source_title=title,
                        content=doc_text,
                        metadata=meta,
                        score=score,
                    )
                )

        return chunks

    def _mock_retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Fallback mock retrieval using keyword matching over seed docs."""
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in ACME_SEED_DOCUMENTS:
            doc_words = set(doc["content"].lower().split())
            overlap = len(query_words.intersection(doc_words))
            score = round(min(1.0, 0.5 + (overlap * 0.1)), 2)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(
                source_type="vector_db",
                source_title=doc["title"],
                content=doc["content"],
                metadata={"category": doc["category"]},
                score=score,
            )
            for score, doc in scored_docs[:top_k]
        ]
