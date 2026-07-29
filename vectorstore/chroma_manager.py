"""ChromaDB Manager Module.

Manages persistent ChromaDB database connections, client lifecycle,
collection creation, and connection status verification.
"""

import logging
from pathlib import Path

from app.config import settings
from vectorstore.collections import ChromaCollectionWrapper

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """Manages connection client and collections for persistent ChromaDB storage."""

    def __init__(self, db_dir: Path | None = None) -> None:
        """Initializes ChromaDB persistent storage client.

        Args:
            db_dir: Directory path for database persistence. Defaults to settings.
        """
        self.db_dir = db_dir or settings.CHROMA_DB_DIR
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

        logger.info("ChromaDBManager initialized with database directory: %s", self.db_dir)

    def _get_client(self):
        """Lazy initialization of persistent ChromaDB Client.

        Returns:
            ChromaDB Client instance.
        """
        if self._client is None:
            logger.info("Connecting to persistent ChromaDB instance at %s...", self.db_dir)
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                self._client = chromadb.PersistentClient(
                    path=str(self.db_dir),
                    settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False),
                )
                logger.info("ChromaDB client connection established successfully.")
            except ImportError:
                logger.warning("ChromaDB package not found. Operating with mock vector client.")
                self._client = "MOCK_CHROMA_CLIENT"
            except Exception as exc:
                logger.error("Failed to connect to ChromaDB: %s", exc, exc_info=True)
                raise
        return self._client

    def get_collection(self, collection_name: str | None = None) -> ChromaCollectionWrapper:
        """Retrieves or creates a named collection inside ChromaDB.

        Args:
            collection_name: Target collection name. Defaults to settings.

        Returns:
            ChromaCollectionWrapper instance encapsulating collection operations.
        """
        name = collection_name or settings.CHROMA_COLLECTION_NAME
        logger.info("Retrieving vector store collection: '%s'", name)
        client = self._get_client()

        if client == "MOCK_CHROMA_CLIENT":
            return ChromaCollectionWrapper(raw_collection=None, name=name, is_mock=True)

        try:
            collection = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Successfully fetched collection: '%s'", name)
            return ChromaCollectionWrapper(raw_collection=collection, name=name)
        except Exception as exc:
            logger.error("Error fetching collection '%s': %s", name, exc, exc_info=True)
            raise

    def is_healthy(self) -> bool:
        """Checks if the vector database client is responsive.

        Returns:
            Boolean True if responsive, False otherwise.
        """
        try:
            client = self._get_client()
            if client == "MOCK_CHROMA_CLIENT":
                return True
            client.heartbeat()
            return True
        except Exception as exc:
            logger.error("ChromaDB health check failed: %s", exc)
            return False
