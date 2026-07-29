"""LLM Response Formatter Module.

Sanitizes raw LLM output, extracts inline citation markers, and standardizes
the response payload structure.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Formats raw model output into structured client responses."""

    def __init__(self) -> None:
        """Initializes the response formatter."""
        logger.info("ResponseFormatter initialized.")

    def format_response(
        self,
        raw_llm_result: Dict[str, Any],
        retrieved_sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Cleans and structures raw LLM output and attached context metadata.

        Args:
            raw_llm_result: Output dict from GeminiLLMClient.
            retrieved_sources: List of source chunk metadata dicts.

        Returns:
            Structured dictionary ready for API response serialization.
        """
        logger.debug("Formatting raw LLM answer payload...")
        answer = raw_llm_result.get("answer", "").strip()

        # Extract cited filenames using regex pattern [Source: filename, Page: N]
        cited_files = set(re.findall(r"\[Source:\s*([^,\]]+)", answer))

        formatted_result = {
            "answer": answer,
            "cited_files": list(cited_files),
            "raw_sources_count": len(retrieved_sources),
        }

        logger.debug("Response formatting complete.")
        return formatted_result
