"""API Pydantic Schemas Module.

Defines request models, response payloads, confidence metrics,
and source citation schemas for FastAPI endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    """Payload model for client RAG query requests."""

    query: str = Field(
        ...,
        description="Machine Learning question or topic prompt.",
        examples=["What is the difference between bagging and boosting in ensemble learning?"],
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve.",
    )
    category_filter: Optional[str] = Field(
        default=None,
        description="Optional domain category filter ('scikit-learn-docs', 'ktu-notes', 'ml-book').",
    )


class SourceDocumentSchema(BaseModel):
    """Schema representing a cited source document."""

    filename: str = Field(..., description="Name of the source file.")
    page_number: int = Field(..., description="Page number of the document.")
    category: str = Field(..., description="Domain category tag.")
    snippet: str = Field(..., description="Text excerpt from the passage.")
    similarity_score: float = Field(..., description="Relevance score [0.0 to 1.0].")


class RAGQueryResponse(BaseModel):
    """Payload model returned to client containing answer, confidence score, and sources."""

    query: str = Field(..., description="Original question string.")
    answer: str = Field(..., description="LLM generated grounded answer text.")
    confidence_score: float = Field(
        ...,
        description="Calculated confidence score between 0.0 and 1.0.",
    )
    sources: List[SourceDocumentSchema] = Field(
        default_factory=list,
        description="List of source document passages supporting the answer.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline execution metrics.",
    )


class IngestRequest(BaseModel):
    """Payload model to trigger ingestion from disk."""

    target_path: Optional[str] = Field(
        default=None,
        description="Local path to file or directory for ingestion. Defaults to raw data folder.",
    )


class IngestResponse(BaseModel):
    """Payload response for ingestion status."""

    status: str = Field(..., description="Status message ('success', 'failed').")
    files_processed: int = Field(..., description="Count of files ingested.")
    total_chunks: int = Field(..., description="Total vector chunks generated.")
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthCheckResponse(BaseModel):
    """Health status endpoint response model."""

    status: str = Field(..., description="Application operational status ('healthy').")
    version: str = Field(..., description="API Version.")
    vectordb_status: str = Field(..., description="ChromaDB connection health.")
