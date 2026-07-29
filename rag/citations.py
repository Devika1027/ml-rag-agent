"""Citation & Source Tracking Module.

Extracts, normalizes, and formats source document metadata into structured
citations for API responses.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class SourceCitation:
    """Dataclass representing a source document citation.

    Attributes:
        filename: Document source filename.
        page_number: Page number in document if available.
        category: Domain category of document.
        snippet: Extracted text snippet preview.
        similarity_score: Similarity score float.
    """

    filename: str
    page_number: int
    category: str
    snippet: str
    similarity_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts citation instance to dictionary representation."""
        return {
            "filename": self.filename,
            "page_number": self.page_number,
            "category": self.category,
            "snippet": self.snippet,
            "similarity_score": self.similarity_score,
        }


class CitationTracker:
    """Tracks and builds structured source document citations."""

    def __init__(self) -> None:
        """Initializes the citation tracker."""
        logger.info("CitationTracker initialized.")

    def build_citations(self, retrieved_chunks: List[Dict[str, Any]]) -> List[SourceCitation]:
        """Builds a deduplicated list of SourceCitation objects from retrieved chunks.

        Args:
            retrieved_chunks: List of chunk dictionaries containing metadata and text.

        Returns:
            List of unique SourceCitation instances.
        """
        logger.debug("Building citations for %d retrieved chunks.", len(retrieved_chunks))
        citations: List[SourceCitation] = []

        seen = set()
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", "Unknown File")
            page = int(meta.get("page_number", 1))
            category = meta.get("category", "ml-general")
            text = chunk.get("text", "")
            score = float(chunk.get("similarity_score", 0.0))

            dedup_key = (filename, page)
            if dedup_key not in seen:
                seen.add(dedup_key)
                citations.append(
                    SourceCitation(
                        filename=filename,
                        page_number=page,
                        category=category,
                        snippet=text[:150] + "..." if len(text) > 150 else text,
                        similarity_score=score,
                    )
                )

        logger.debug("Generated %d unique document citations.", len(citations))
        return citations
