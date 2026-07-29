"""Ingestion Manager Orchestrator Module.

Coordinates document loading, metadata extraction, text sanitization,
section parsing, chunking, embedding generation, and vector store persistence.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings
from ingestion.cleaner import TextCleaner
from ingestion.chunker import DocumentChunk, DocumentChunker
from ingestion.embedder import EmbeddingGenerator
from ingestion.extractor import MetadataExtractor
from ingestion.loader import DocumentLoader
from vectorstore.chroma_manager import ChromaDBManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end document ingestion pipeline."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        extractor: MetadataExtractor | None = None,
        cleaner: TextCleaner | None = None,
        chunker: DocumentChunker | None = None,
        embedder: EmbeddingGenerator | None = None,
        vectorstore: ChromaDBManager | None = None,
    ) -> None:
        """Initializes the ingestion pipeline components.

        Uses dependency injection to accept custom component overrides.
        """
        self.loader = loader or DocumentLoader()
        self.extractor = extractor or MetadataExtractor()
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or DocumentChunker()
        self.embedder = embedder or EmbeddingGenerator()
        self.vectorstore = vectorstore or ChromaDBManager()

        logger.info("IngestionPipeline initialized with all required sub-modules.")

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Ingests a single file into the vector database.

        Args:
            file_path: Path to target document file.

        Returns:
            Summary dictionary of ingestion metrics (chunks_processed, collection_name).
        """
        logger.info("Starting ingestion workflow for file: %s", file_path)

        # 1. Load Document
        loaded_docs = self.loader.load_file(file_path)

        all_chunks: List[DocumentChunk] = []
        for doc in loaded_docs:
            # 2. Extract Metadata
            meta = self.extractor.extract(doc)

            # 3. Clean Text
            cleaned_text = self.cleaner.clean(doc.content)

            # 4. Chunk Document
            doc_chunks = self.chunker.split_text(cleaned_text, base_metadata=meta)
            all_chunks.extend(doc_chunks)

        if not all_chunks:
            logger.warning("No valid chunks produced for file: %s", file_path)
            return {"status": "warning", "chunks_processed": 0, "file": str(file_path)}

        # 5. Generate Embeddings
        chunk_texts = [c.text for c in all_chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)

        # 6. Store in ChromaDB
        collection = self.vectorstore.get_collection(settings.CHROMA_COLLECTION_NAME)
        collection.add(
            ids=[c.chunk_id for c in all_chunks],
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=[c.metadata for c in all_chunks],
        )

        logger.info(
            "Successfully ingested file %s (%d chunks stored into ChromaDB).",
            file_path.name,
            len(all_chunks),
        )

        return {
            "status": "success",
            "chunks_processed": len(all_chunks),
            "file": str(file_path),
            "collection": settings.CHROMA_COLLECTION_NAME,
        }

    def process_directory(self, directory_path: Path) -> Dict[str, Any]:
        """Ingests an entire directory of documents into the vector store.

        Args:
            directory_path: Directory containing documents.

        Returns:
            Aggregate ingestion metrics dictionary.
        """
        logger.info("Starting batch directory ingestion workflow for: %s", directory_path)
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        processed_files = 0
        total_chunks = 0

        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.loader.supported_extensions:
                try:
                    result = self.process_file(file_path)
                    processed_files += 1
                    total_chunks += result.get("chunks_processed", 0)
                except Exception as exc:
                    logger.error("Failed to ingest file %s: %s", file_path, exc)

        logger.info(
            "Directory ingestion completed. Files: %d, Total Chunks: %d",
            processed_files,
            total_chunks,
        )

        return {
            "status": "success",
            "files_processed": processed_files,
            "total_chunks": total_chunks,
            "directory": str(directory_path),
        }
