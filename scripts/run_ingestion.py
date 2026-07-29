"""Standalone Script: Run Document Ingestion.

Executes document loading, text splitting, embedding generation,
and ChromaDB persistence for documents located in raw data directory.
"""

import logging
from pathlib import Path

from app.config import RAW_DATA_DIR, setup_logging
from ingestion.ingest import IngestionPipeline

setup_logging("INFO")
logger = logging.getLogger(__name__)


def main():
    """Executes document ingestion script."""
    logger.info("Initializing standalone document ingestion script...")

    target_dir = RAW_DATA_DIR
    logger.info("Ingestion target directory: %s", target_dir)

    pipeline = IngestionPipeline()

    if not target_dir.exists() or not any(target_dir.iterdir()):
        logger.warning(
            "Raw data directory '%s' is empty or does not exist. "
            "Please place Machine Learning PDF files into '%s' before running ingestion.",
            target_dir,
            target_dir,
        )
        return

    result = pipeline.process_directory(target_dir)
    logger.info("Ingestion completed successfully. Summary: %s", result)


if __name__ == "__main__":
    main()
