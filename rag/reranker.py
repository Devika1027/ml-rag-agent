"""Reranker Module.

Applies cross-encoder scoring to re-rank and re-order retrieved context chunks
for maximum relevance prior to LLM generation.
"""

import logging
from typing import Any, Dict, List

from app.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieved candidate chunks using deep semantic cross-encoders."""

    def __init__(self, enabled: bool | None = None) -> None:
        """Initializes the reranker.

        Args:
            enabled: Toggle reranking execution. Defaults to settings.
        """
        self.enabled = enabled if enabled is not None else settings.ENABLE_RERANKER
        logger.info("CrossEncoderReranker initialized (enabled=%s)", self.enabled)

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reranks candidate chunks based on cross-encoder query-passage relevance scores.

        Args:
            query: User query question string.
            candidate_chunks: Initial list of retrieved context chunk dicts.

        Returns:
            Re-ordered list of chunk dicts sorted by descending relevance score.
        """
        if not self.enabled or not candidate_chunks:
            logger.debug("Reranker disabled or empty candidate list. Returning original list.")
            return candidate_chunks

        logger.info("Reranking %d candidate chunks for query: '%s'", len(candidate_chunks), query)

        # Placeholder: Integrate SentenceTransformers CrossEncoder model scoring
        # For starter: sort using existing similarity_score
        reranked = sorted(
            candidate_chunks,
            key=lambda x: x.get("similarity_score", 0.0),
            reverse=True,
        )

        logger.info("Reranking complete.")
        return reranked
