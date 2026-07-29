"""Text Cleaner Module.

Normalizes extracted document text, cleans artifacts, removes headers/footers,
and formats whitespace for high-quality embedding generation.
"""

import logging
import re

logger = logging.getLogger(__name__)


class TextCleaner:
    """Cleans and sanitizes document text for downstream NLP processing."""

    def __init__(self) -> None:
        """Initializes regex patterns for text sanitization."""
        logger.info("Initializing TextCleaner.")

    def clean(self, text: str) -> str:
        """Applies sanitization steps to raw input text.

        Args:
            text: Raw string extracted from document.

        Returns:
            Cleaned and normalized text string.
        """
        if not text:
            return ""

        logger.debug("Cleaning text snippet of length: %d", len(text))
        cleaned_text = text

        # Replace excessive whitespace and newlines
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

        # Remove common PDF page headers/footers patterns
        cleaned_text = re.sub(r"Page \d+ of \d+", "", cleaned_text, flags=re.IGNORECASE)

        # Strip surrounding whitespace
        cleaned_text = cleaned_text.strip()

        logger.debug("Text cleaned successfully. New length: %d", len(cleaned_text))
        return cleaned_text
