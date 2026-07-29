"""Unit tests for Vector Store Retriever module."""

from rag.retriever import VectorRetriever


def test_retriever_initialization():
    """Verify default parameters of vector retriever."""
    retriever = VectorRetriever(top_k=3)
    assert retriever.top_k == 3
    assert retriever.embedder is not None
    assert retriever.vectorstore is not None


def test_retriever_mock_query():
    """Test retrieving matching documents from mock vectorstore."""
    retriever = VectorRetriever(top_k=2)
    results = retriever.retrieve("What is supervised learning?")

    assert isinstance(results, list)
    assert len(results) > 0
    assert "text" in results[0]
    assert "similarity_score" in results[0]
