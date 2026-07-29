"""Metadata Extractor Module.

Extracts structured metadata properties from loaded document contents and file attributes.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from ingestion.loader import LoadedDocument

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts domain metadata (title, subject, category, author) from ML documents."""

    def __init__(self) -> None:
        """Initializes the metadata extractor."""
        logger.info("MetadataExtractor initialized.")

    def extract(self, doc: LoadedDocument) -> Dict[str, Any]:
        """Extracts metadata attributes from a loaded document.

        Args:
            doc: LoadedDocument instance.

        Returns:
            Dictionary containing extracted metadata fields.
        """
        logger.debug("Extracting metadata for file: %s", doc.source_path.name)
        metadata: Dict[str, Any] = {
            "source": str(doc.source_path),
            "filename": doc.source_path.name,
            "file_type": doc.source_path.suffix.lstrip("."),
            "page_number": doc.page_number or 1,
            "category": self._infer_category(doc.source_path),
        }

        # Merge pre-existing document metadata
        metadata.update(doc.metadata)
        logger.debug("Extracted metadata keys: %s", list(metadata.keys()))
        return metadata

    def _infer_category(self, file_path: Path) -> str:
        """Infers domain category based on file directory path or name.

        Args:
            file_path: Path object of document.

        Returns:
            Category tag string (e.g., 'scikit-learn', 'ktu-notes', 'ml-paper').
        """
        path_str = str(file_path).lower()
        if "scikit" in path_str or "sklearn" in path_str:
            return "scikit-learn-docs"
        elif "ktu" in path_str:
            return "ktu-notes"
        elif "book" in path_str:
            return "ml-book"
        return "ml-general"
