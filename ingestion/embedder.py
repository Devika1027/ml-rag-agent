"""Sentence Transformer Embedder Module.

Generates dense vector embeddings using HuggingFace Sentence Transformers
for document chunks and user queries.
"""

import logging
from typing import List

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Wrapper around HuggingFace SentenceTransformers embedding model."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        """Initializes the embedding model.

        Args:
            model_name: HuggingFace model identifier. Defaults to settings.
            device: Target execution device ('cpu' or 'cuda').
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE
        self._model = None
        logger.info(
            "Initializing EmbeddingGenerator with model: %s on device: %s",
            self.model_name,
            self.device,
        )

    def _get_model(self):
        """Lazy loader for SentenceTransformer model instance.

        Returns:
            SentenceTransformer model object.
        """
        if self._model is None:
            logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info("SentenceTransformer model loaded successfully.")
            except ImportError:
                logger.warning("SentenceTransformers library not installed. Falling back to mock embedder.")
                self._model = "MOCK_MODEL"
            except Exception as exc:
                logger.error("Failed to load SentenceTransformer model: %s", exc, exc_info=True)
                raise
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a single text string.

        Args:
            text: Input text string.

        Returns:
            List of floating-point values representing embedding vector.
        """
        logger.debug("Generating embedding vector for text snippet of length: %d", len(text))
        model = self._get_model()

        if model == "MOCK_MODEL":
            # Mock 384-dimensional vector for testing/fallback
            return [0.01 * (i % 10) for i in range(384)]

        embeddings = model.encode(text, convert_to_numpy=True).tolist()
        return embeddings

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a batch of text strings.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """
        logger.info("Generating embeddings for batch of %d text items.", len(texts))
        if not texts:
            return []

        model = self._get_model()
        if model == "MOCK_MODEL":
            return [[0.01 * (i % 10) for i in range(384)] for _ in texts]

        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        logger.info("Successfully generated %d batch embeddings.", len(embeddings))
        return embeddings
