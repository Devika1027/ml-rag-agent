"""FastAPI Route Handlers Module.

Defines HTTP REST endpoints for health checks, RAG queries,
and document ingestion triggers.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_ingestion_pipeline, get_rag_pipeline, get_settings
from api.schemas import (
    HealthCheckResponse,
    IngestRequest,
    IngestResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    SourceDocumentSchema,
)
from app.config import RAW_DATA_DIR, Settings
from ingestion.ingest import IngestionPipeline
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["RAG Services"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check status",
    description="Returns application health status and vector database state.",
)
async def health_check(
    rag: RAGPipeline = Depends(get_rag_pipeline),
) -> HealthCheckResponse:
    """Verifies service health and ChromaDB connectivity."""
    db_healthy = rag.retriever.vectorstore.is_healthy()
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        vectordb_status="connected" if db_healthy else "degraded",
    )


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Submit RAG Query",
    description="Processes user questions, retrieves grounded ML context, and returns answer with confidence score.",
)
async def query_rag(
    request: RAGQueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> RAGQueryResponse:
    """Handles client RAG query requests."""
    logger.info("Received API RAG Query: '%s'", request.query)
    try:
        result = rag_pipeline.run(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category_filter,
        )

        sources_schema = [
            SourceDocumentSchema(
                filename=s.filename,
                page_number=s.page_number,
                category=s.category,
                snippet=s.snippet,
                similarity_score=s.similarity_score,
            )
            for s in result.sources
        ]

        return RAGQueryResponse(
            query=result.query,
            answer=result.answer,
            confidence_score=result.confidence_score,
            sources=sources_schema,
            metadata=result.metadata,
        )

    except Exception as exc:
        logger.error("Error processing RAG query endpoint: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the RAG query: {str(exc)}",
        )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Trigger Document Ingestion",
    description="Triggers reading, splitting, embedding, and indexing of ML documents.",
)
async def trigger_ingestion(
    request: IngestRequest,
    ingest_pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    app_settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Triggers batch document ingestion."""
    target_path = Path(request.target_path) if request.target_path else RAW_DATA_DIR
    logger.info("Received Ingestion request for path: %s", target_path)

    if not target_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target ingestion path not found: {target_path}",
        )

    try:
        if target_path.is_file():
            res = ingest_pipeline.process_file(target_path)
            return IngestResponse(
                status="success",
                files_processed=1,
                total_chunks=res.get("chunks_processed", 0),
                details=res,
            )
        else:
            res = ingest_pipeline.process_directory(target_path)
            return IngestResponse(
                status="success",
                files_processed=res.get("files_processed", 0),
                total_chunks=res.get("total_chunks", 0),
                details=res,
            )
    except Exception as exc:
        logger.error("Error during ingestion trigger: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(exc)}",
        )
