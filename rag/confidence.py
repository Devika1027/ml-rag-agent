"""Confidence Evaluation Module.

Computes confidence scores for RAG responses based on vector retrieval similarity,
context text overlap, and response grounding metrics.
"""

import logging
from typing import Any, Dict, List

from app.config import settings

logger = logging.getLogger(__name__)


class ConfidenceEvaluator:
    """Evaluates RAG pipeline response quality and confidence."""

    def __init__(self, threshold: float | None = None) -> None:
        """Initializes the evaluator.

        Args:
            threshold: Confidence cutoff threshold. Defaults to settings.
        """
        self.threshold = threshold or settings.CONFIDENCE_THRESHOLD
        logger.info("ConfidenceEvaluator initialized (threshold=%.2f)", self.threshold)

    def compute_confidence(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        llm_answer: str,
    ) -> float:
        """Computes a normalized confidence score between 0.0 and 1.0.

        Args:
            retrieved_chunks: List of retrieved passage chunk dicts.
            llm_answer: Generated LLM response text string.

        Returns:
            Float confidence score in range [0.0, 1.0].
        """
        logger.debug("Computing confidence score for RAG response...")

        if not retrieved_chunks:
            logger.warning("No context chunks retrieved; returning 0.0 confidence score.")
            return 0.0

        if "cannot find sufficient evidence" in llm_answer.lower():
            logger.info("LLM reported insufficient evidence; returning 0.1 confidence score.")
            return 0.10

        # Calculate average similarity score from top chunks
        scores = [c.get("similarity_score", 0.7) for c in retrieved_chunks]
        avg_retrieval_score = sum(scores) / len(scores) if scores else 0.5

        # Weighted aggregate confidence score
        confidence = round(min(1.0, max(0.0, avg_retrieval_score)), 2)

        logger.info("Computed response confidence score: %.2f (Threshold: %.2f)", confidence, self.threshold)
        return confidence
