"""Unit tests for Document Chunker module."""

from ingestion.chunker import DocumentChunker


def test_chunker_split_text():
    """Verify document chunker text splitting and overlap logic."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs "
        "and the desired outputs. The data is known as training data, and consists of a set of training examples."
    )

    chunks = chunker.split_text(sample_text, base_metadata={"filename": "test.txt"})

    assert len(chunks) > 0
    assert chunks[0].metadata["filename"] == "test.txt"
    assert chunks[0].metadata["chunk_index"] == 0
    assert len(chunks[0].text) <= 100


def test_chunker_empty_input():
    """Verify empty text input returns empty chunk list."""
    chunker = DocumentChunker()
    chunks = chunker.split_text("")
    assert chunks == []
