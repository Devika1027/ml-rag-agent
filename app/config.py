"""Application Configuration Module.

Centralizes configuration settings using Pydantic Settings, pathlib,
and manages logging setup across the application.
"""

import logging
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
VECTORDB_DIR: Path = DATA_DIR / "vectordb"


class Settings(BaseSettings):
    """Application settings schema backed by environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General App Config
    PROJECT_NAME: str = "Production Machine Learning RAG Agent"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: str = Field(default="INFO", description="Logging output level")

    # FastAPI Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Google Gemini Settings
    GEMINI_API_KEY: str = Field(default="", description="API Key for Google Gemini")
    GEMINI_MODEL_NAME: str = Field(
        default="gemini-1.5-flash",
        description="Google Gemini LLM model identifier",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.2, description="Sampling temperature for LLM text generation"
    )
    LLM_MAX_OUTPUT_TOKENS: int = Field(
        default=2048, description="Maximum token length for generated answers"
    )

    # Embedding Settings
    EMBEDDING_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace Sentence Transformers model name",
    )
    EMBEDDING_DEVICE: str = Field(default="cpu", description="Target execution device for embeddings")

    # Vector Database Settings
    CHROMA_DB_DIR: Path = Field(
        default=VECTORDB_DIR,
        description="Local filesystem directory for ChromaDB persistence",
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="ml_documents",
        description="Default Chroma collection name",
    )

    # Document Chunking Settings
    CHUNK_SIZE: int = Field(default=500, description="Target character chunk size")
    CHUNK_OVERLAP: int = Field(default=50, description="Overlapping characters between chunks")

    # RAG Retrieval Settings
    RETRIEVAL_TOP_K: int = Field(
        default=5, description="Number of top document chunks to retrieve"
    )
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.65, description="Minimum acceptable similarity confidence score"
    )
    ENABLE_RERANKER: bool = Field(
        default=False, description="Whether to apply cross-encoder reranking"
    )

    # Automation Webhook Secret
    AUTOMATION_WEBHOOK_SECRET: str = Field(
        default="secret", description="Secret token for verifying n8n automation webhooks"
    )


def setup_logging(level_name: str = "INFO") -> None:
    """Configures application-wide logging handlers and formatters.

    Args:
        level_name: Log level name (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    log_level = getattr(logging, level_name.upper(), logging.INFO)
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


# Singleton settings instance
settings = Settings()
setup_logging(settings.LOG_LEVEL)
