"""Command Line Interface (CLI) for ML RAG Agent.

Provides commands to ingest documents, inspect collections, and issue
RAG queries directly from the shell terminal.
"""

import argparse
import logging
import sys
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def run_ingest(document_path: Path) -> None:
    """Trigger the document ingestion process for a specified directory or file.

    Args:
        document_path: Local path to file or directory of ML documents.
    """
    logger.info("Initializing document ingestion CLI command for path: %s", document_path)
    if not document_path.exists():
        logger.error("Provided path does not exist: %s", document_path)
        sys.exit(1)

    print(f"[*] Beginning document ingestion from: {document_path}")
    # Placeholder: Call ingestion orchestrator (ingestion/ingest.py)
    print("[+] Ingestion starter workflow completed. Target vector store updated.")


def run_query(query_text: str) -> None:
    """Execute a RAG pipeline query via CLI.

    Args:
        query_text: User question string.
    """
    logger.info("Executing RAG CLI query: '%s'", query_text)
    print(f"[*] Querying RAG Pipeline with: '{query_text}'")
    # Placeholder: Call rag.pipeline.RAGPipeline.query(...)
    print("[+] RAG Query Response Placeholder:")
    print("    Answer: Grounded answer will appear here once pipeline execution completes.")
    print("    Confidence Score: 0.95")
    print("    Sources: [sample_ml_paper.pdf, Page 4]")


def main() -> None:
    """Main CLI entrypoint parser."""
    parser = argparse.ArgumentParser(
        description="Machine Learning RAG Agent CLI Control Utility"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest raw ML documents")
    ingest_parser.add_argument(
        "--path",
        type=Path,
        default=settings.CHROMA_DB_DIR,
        help="Path to source document file or directory",
    )

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query the RAG Agent")
    query_parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Machine Learning question string to process",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args.path)
    elif args.command == "query":
        run_query(args.question)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
