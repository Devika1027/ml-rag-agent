"""Document Loader Module.

Handles reading PDF, Markdown, and text files from local storage into structured
in-memory representations containing raw content and source attributes.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    """Dataclass representing an ingested raw document page or file.

    Attributes:
        content: Extracted text content of the document/page.
        source_path: Path to the original document on disk.
        page_number: Optional 1-indexed page number if sourced from PDF.
        metadata: Key-value dictionary storing metadata attributes.
    """

    content: str
    source_path: Path
    page_number: int | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Reads Machine Learning source files (PDFs, TXT, MD) and extracts raw text."""

    def __init__(self, supported_extensions: List[str] | None = None) -> None:
        """Initializes the document loader with allowed file extensions.

        Args:
            supported_extensions: List of file extensions to process.
        """
        self.supported_extensions = supported_extensions or [".pdf", ".txt", ".md"]
        logger.info("Initialized DocumentLoader with extensions: %s", self.supported_extensions)

    def load_file(self, file_path: Path) -> List[LoadedDocument]:
        """Loads a single document file from disk.

        Args:
            file_path: Absolute or relative Path to target document.

        Returns:
            List of LoadedDocument objects for each page/section.

        Raises:
            FileNotFoundError: If file_path does not exist on disk.
            ValueError: If file extension is unsupported.
        """
        logger.info("Loading document file: %s", file_path)
        if not file_path.exists():
            logger.error("Document file not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in self.supported_extensions:
            logger.error("Unsupported file extension '%s' for file: %s", ext, file_path)
            raise ValueError(f"Unsupported format: {ext}")

        documents: List[LoadedDocument] = []

        if ext == ".pdf":
            documents = self._load_pdf(file_path)
        else:
            documents = self._load_text(file_path)

        logger.info("Successfully loaded %d document objects from %s", len(documents), file_path)
        return documents

    def load_directory(self, directory_path: Path) -> List[LoadedDocument]:
        """Recursively scans a directory and loads all supported ML document files.

        Args:
            directory_path: Directory path to scan.

        Returns:
            Combined list of all loaded documents across files.
        """
        logger.info("Scanning directory for documents: %s", directory_path)
        if not directory_path.is_dir():
            logger.error("Path is not a valid directory: %s", directory_path)
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        all_docs: List[LoadedDocument] = []
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    docs = self.load_file(file_path)
                    all_docs.extend(docs)
                except Exception as exc:
                    logger.warning("Failed to load file %s: %s", file_path, exc)

        logger.info("Directory scan completed. Total loaded pages/files: %d", len(all_docs))
        return all_docs

    def _load_pdf(self, file_path: Path) -> List[LoadedDocument]:
        """Internal helper to parse PDF files page-by-page using pypdf or pdfplumber.

        Args:
            file_path: Path to PDF file.

        Returns:
            List of LoadedDocument objects per PDF page.
        """
        logger.debug("Reading PDF document: %s", file_path)
        documents: List[LoadedDocument] = []

        # Try pypdf first
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)
            logger.info("Extracting text from %d pages in PDF %s", total_pages, file_path.name)

            for idx, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        LoadedDocument(
                            content=text,
                            source_path=file_path,
                            page_number=idx,
                            metadata={"filename": file_path.name, "total_pages": total_pages},
                        )
                    )
            if documents:
                return documents
        except Exception as exc:
            logger.warning("pypdf parsing failed for %s (%s). Attempting pdfplumber fallback...", file_path.name, exc)

        # Fallback to pdfplumber
        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for idx, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append(
                            LoadedDocument(
                                content=text,
                                source_path=file_path,
                                page_number=idx,
                                metadata={"filename": file_path.name, "total_pages": total_pages},
                            )
                        )
        except Exception as exc:
            logger.error("pdfplumber parsing failed for %s: %s", file_path.name, exc, exc_info=True)

        return documents

    def _load_text(self, file_path: Path) -> List[LoadedDocument]:
        """Internal helper to load standard plain text / Markdown files.

        Args:
            file_path: Path to text file.

        Returns:
            List containing a single LoadedDocument object.
        """
        logger.debug("Reading text file: %s", file_path)
        try:
            content = file_path.read_text(encoding="utf-8")
            return [
                LoadedDocument(
                    content=content,
                    source_path=file_path,
                    page_number=1,
                    metadata={"filename": file_path.name},
                )
            ]
        except Exception as exc:
            logger.error("Error reading text file %s: %s", file_path, exc, exc_info=True)
            raise
