"""ChromaDB Collection Operations Wrapper Module.

Provides standard higher-level methods for vector collection management,
insertions, vector similarity queries, and metadata filtering.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChromaCollectionWrapper:
    """Wrapper encapsulating ChromaDB Collection vector operations."""

    def __init__(self, raw_collection: Any, name: str, is_mock: bool = False) -> None:
        """Initializes the collection wrapper.

        Args:
            raw_collection: Direct ChromaDB collection handle.
            name: Collection identifier name.
            is_mock: Flag indicating whether underlying client is mocked.
        """
        self.collection = raw_collection
        self.name = name
        self.is_mock = is_mock
        logger.info("ChromaCollectionWrapper initialized for collection '%s'", name)

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Inserts or updates vector records in the collection.

        Args:
            ids: List of unique string chunk IDs.
            embeddings: List of embedding vectors matching chunks.
            documents: List of text chunk content strings.
            metadatas: Optional list of metadata dictionaries.
        """
        logger.info("Inserting %d records into collection '%s'.", len(ids), self.name)
        if self.is_mock:
            logger.debug("Mock add operation called for %d records.", len(ids))
            return

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("Successfully added %d vectors to collection '%s'.", len(ids), self.name)
        except Exception as exc:
            logger.error("Failed inserting vectors to Chroma collection: %s", exc, exc_info=True)
            raise

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Queries collection vectors by similarity to query embeddings.

        Args:
            query_embeddings: List containing query embedding vector(s).
            n_results: Number of nearest matches to return per query.
            where_filter: Metadata filter constraint dictionary.

        Returns:
            QueryResult dictionary with ids, distances, documents, and metadatas.
        """
        logger.info("Querying collection '%s' for top %d matches.", self.name, n_results)
        if self.is_mock:
            logger.debug("Mock vector query executed.")
            return {
                "ids": [["mock_chunk_1", "mock_chunk_2"]],
                "distances": [[0.15, 0.25]],
                "documents": [
                    [
                        "Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs.",
                        "Scikit-learn provides algorithms like Random Forest, Support Vector Machines, and Gradient Boosting.",
                    ]
                ],
                "metadatas": [
                    [
                        {"filename": "supervised_learning.pdf", "page_number": 3},
                        {"filename": "sklearn_guide.pdf", "page_number": 12},
                    ]
                ],
            }

        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where_filter,
            )
            logger.info("Query executed successfully. Found %d result sets.", len(results.get("ids", [])))
            return results
        except Exception as exc:
            logger.error("Error executing vector query on collection '%s': %s", self.name, exc, exc_info=True)
            raise

    def count(self) -> int:
        """Returns the total number of items stored in the collection.

        Returns:
            Integer total vector count.
        """
        if self.is_mock:
            return 42

        try:
            return self.collection.count()
        except Exception as exc:
            logger.error("Error getting collection count: %s", exc)
            return 0
