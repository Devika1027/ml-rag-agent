"""Unit tests for Document Loader module."""

import pytest
from pathlib import Path

from ingestion.loader import DocumentLoader, LoadedDocument
from ingestion.extractor import MetadataExtractor


def test_loader_supported_extensions():
    """Verify default supported file extensions in loader."""
    loader = DocumentLoader()
    assert ".pdf" in loader.supported_extensions
    assert ".txt" in loader.supported_extensions
    assert ".md" in loader.supported_extensions


def test_loader_file_not_found():
    """Verify FileNotFoundError raised for non-existent paths."""
    loader = DocumentLoader()
    non_existent = Path("data/raw/does_not_exist.pdf")
    with pytest.raises(FileNotFoundError):
        loader.load_file(non_existent)


def test_loader_text_file(tmp_path):
    """Test reading plain text document."""
    test_file = tmp_path / "sample_ml.txt"
    test_file.write_text("Gradient boosting is an ensemble machine learning technique.", encoding="utf-8")

    loader = DocumentLoader()
    docs = loader.load_file(test_file)

    assert len(docs) == 1
    assert "Gradient boosting" in docs[0].content
    assert docs[0].source_path == test_file


def test_metadata_extractor(tmp_path):
    """Test metadata extraction from loaded document."""
    test_file = tmp_path / "ktu_notes_ml.txt"
    doc = LoadedDocument(
        content="Module 1: Decision Trees",
        source_path=test_file,
        page_number=1,
    )
    extractor = MetadataExtractor()
    meta = extractor.extract(doc)

    assert meta["filename"] == "ktu_notes_ml.txt"
    assert meta["category"] == "ktu-notes"
    assert meta["page_number"] == 1
