"""Intelligent Document Chunker Module.

Splits document text into optimal context windows with overlapping boundaries
to preserve semantics for vector embedding generation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a text chunk ready for vector embedding and database insertion.

    Attributes:
        chunk_id: Unique string identifier for the chunk.
        text: Text content of the chunk.
        metadata: Associated document and chunk metadata.
    """

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """Splits document text into overlapping chunks using character or token metrics."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Initializes chunking parameters.

        Args:
            chunk_size: Target size in characters/tokens. Defaults to app settings.
            chunk_overlap: Overlap size between chunks. Defaults to app settings.
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        logger.info(
            "Initialized DocumentChunker (chunk_size=%d, chunk_overlap=%d)",
            self.chunk_size,
            self.chunk_overlap,
        )

    def split_text(self, text: str, base_metadata: Dict[str, Any] | None = None) -> List[DocumentChunk]:
        """Splits raw text into a list of DocumentChunk objects.

        Args:
            text: Sanitized text content.
            base_metadata: Inherited document metadata.

        Returns:
            List of generated DocumentChunk instances.
        """
        if not text:
            return []

        base_meta = base_metadata.copy() if base_metadata else {}
        chunks: List[DocumentChunk] = []
        doc_filename = base_meta.get("filename", "doc")

        logger.debug("Splitting text of length %d into chunks...", len(text))
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunk_meta = base_meta.copy()
            chunk_meta["chunk_index"] = chunk_index
            chunk_id = f"{doc_filename}_chunk_{chunk_index}"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=chunk_meta,
                )
            )

            chunk_index += 1
            start += self.chunk_size - self.chunk_overlap

        logger.info("Successfully generated %d chunks for document: %s", len(chunks), doc_filename)
        return chunks
