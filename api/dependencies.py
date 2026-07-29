"""FastAPI Dependency Injection Module.

Provides reusable dependency providers for Settings, RAG Pipeline,
and Ingestion services across API routes.
"""

import logging
from typing import Generator

from app.config import Settings, settings
from ingestion.ingest import IngestionPipeline
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Global singletons for performance efficiency
_rag_pipeline_instance: RAGPipeline | None = None
_ingestion_pipeline_instance: IngestionPipeline | None = None


def get_settings() -> Settings:
    """Dependency provider for application settings.

    Returns:
        Settings singleton instance.
    """
    return settings


def get_rag_pipeline() -> RAGPipeline:
    """Dependency provider for RAG Pipeline.

    Returns:
        RAGPipeline singleton instance.
    """
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        logger.info("Instantiating RAGPipeline dependency singleton...")
        _rag_pipeline_instance = RAGPipeline()
    return _rag_pipeline_instance


def get_ingestion_pipeline() -> IngestionPipeline:
    """Dependency provider for Ingestion Pipeline.

    Returns:
        IngestionPipeline singleton instance.
    """
    global _ingestion_pipeline_instance
    if _ingestion_pipeline_instance is None:
        logger.info("Instantiating IngestionPipeline dependency singleton...")
        _ingestion_pipeline_instance = IngestionPipeline()
    return _ingestion_pipeline_instance
