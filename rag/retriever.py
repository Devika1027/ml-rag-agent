"""Vector Store Retriever Module.

Fetches relevant document context chunks from ChromaDB for a given query embedding,
with optional metadata filtering.
"""

import logging
from typing import Any, Dict, List, Optional

from app.config import settings
from ingestion.embedder import EmbeddingGenerator
from vectorstore.chroma_manager import ChromaDBManager

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retrieves context chunks from ChromaDB using dense vector search."""

    def __init__(
        self,
        embedder: Optional[EmbeddingGenerator] = None,
        vectorstore: Optional[ChromaDBManager] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """Initializes the retriever with embedding and vectorstore components.

        Args:
            embedder: EmbeddingGenerator instance. Defaults to new instance.
            vectorstore: ChromaDBManager instance. Defaults to new instance.
            top_k: Number of top documents to retrieve. Defaults to settings.
        """
        self.embedder = embedder or EmbeddingGenerator()
        self.vectorstore = vectorstore or ChromaDBManager()
        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        logger.info("VectorRetriever initialized with top_k=%d", self.top_k)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves top matching text chunks for a query string.

        Args:
            query: User query question string.
            top_k: Override for top K document count.
            category_filter: Optional category tag filter (e.g. 'scikit-learn-docs').

        Returns:
            List of dictionaries containing chunk text, metadata, similarity distance, and chunk ID.
        """
        k = top_k or self.top_k
        logger.info("Executing retrieval query: '%s' (top_k=%d)", query, k)

        # 1. Embed Query
        query_embedding = self.embedder.embed_text(query)

        # 2. Build metadata filter if specified
        where_filter = {"category": category_filter} if category_filter else None

        # 3. Query ChromaDB Collection
        collection = self.vectorstore.get_collection()
        raw_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where_filter=where_filter,
        )

        retrieved_chunks: List[Dict[str, Any]] = []

        # Parse raw Chroma output arrays
        if raw_results and "documents" in raw_results and raw_results["documents"]:
            docs = raw_results["documents"][0]
            metas = raw_results.get("metadatas", [[]])[0]
            dists = raw_results.get("distances", [[]])[0]
            ids = raw_results.get("ids", [[]])[0]

            for idx in range(len(docs)):
                distance = dists[idx] if idx < len(dists) else 0.0
                # Cosine similarity score approximation from cosine distance
                similarity_score = max(0.0, 1.0 - distance)

                retrieved_chunks.append(
                    {
                        "chunk_id": ids[idx] if idx < len(ids) else f"chunk_{idx}",
                        "text": docs[idx],
                        "metadata": metas[idx] if idx < len(metas) else {},
                        "distance": distance,
                        "similarity_score": round(similarity_score, 4),
                    }
                )

        logger.info("Retrieved %d matching context chunks.", len(retrieved_chunks))
        return retrieved_chunks
