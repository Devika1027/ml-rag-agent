# Production Machine Learning RAG Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.0+-orange.svg)](https://www.trychroma.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-API-8E44AD.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, production-ready **Retrieval-Augmented Generation (RAG) Agent** built in **Python 3.12+** following **Clean Architecture** software engineering principles.

This application ingests technical Machine Learning documents (PDF papers, Scikit-Learn documentation, KTU lecture notes, and textbooks), extracts text and metadata, constructs overlapping vector chunks, persists embeddings in ChromaDB using HuggingFace Sentence Transformers, retrieves grounded context, generates grounded answers using Google Gemini LLM API, computes confidence scores, and returns citations via a FastAPI service.

---

## Key Features

- **Clean Architecture**: Strict separation of concerns (API, App Config, Ingestion, VectorStore, RAG Engine, LLM, Automation, Tests).
- **Intelligent Chunking & Ingestion**: Robust text cleaning, metadata extraction, section parsing, and overlapping chunking.
- **Dense Vector Search**: Powered by HuggingFace `all-MiniLM-L6-v2` embeddings and persistent **ChromaDB**.
- **Zero-Hallucination Grounded Answers**: Integrated with **Google Gemini 1.5** using strict system prompt grounding.
- **Confidence Scoring & Citations**: Computes confidence scores ($0.0$ to $1.0$) and tracks exact source document names and page numbers.
- **Automation Ready**: Pre-configured email processor and n8n JSON workflow integration.
- **Production API**: Built with FastAPI, Pydantic v2, dependency injection, and comprehensive OpenAPI documentation.

---

## Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                                FastAPI Client Layer                               |
|                  POST /api/v1/query   |   POST /api/v1/ingest                     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                 RAG Pipeline Engine                               |
|              (rag/pipeline.py, rag/confidence.py, rag/citations.py)              |
+-----------------------------------------------------------------------------------+
                   |                                             |
                   v                                             v
+------------------------------------+         +------------------------------------+
|        Ingestion & VectorStore     |         |             LLM Engine             |
|   (ingestion/, vectorstore/)       |         |        (llm/gemini.py,             |
|  - SentenceTransformer Embeddings  |         |         llm/prompts.py)            |
|  - ChromaDB Vector Store Persistence|        |  - Google Gemini 1.5 Flash API     |
+------------------------------------+         +------------------------------------+
```

---

## Required Folder Structure

```
ml-rag-agent/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entrypoint & lifespan
│   ├── routes.py                  # API endpoints (/health, /query, /ingest)
│   ├── schemas.py                 # Pydantic v2 request/response schemas
│   └── dependencies.py            # FastAPI dependency injection providers
├── app/
│   ├── __init__.py
│   ├── cli.py                     # CLI interface for ingestion & queries
│   └── config.py                  # Pydantic Settings & logging setup
├── automation/
│   ├── __init__.py
│   ├── email/
│   │   ├── __init__.py
│   │   └── processor.py           # Email RAG response payload generator
│   └── n8n/
│       └── workflow_template.json # n8n workflow integration schema
├── data/
│   ├── raw/                       # Raw source ML PDFs & text files
│   ├── processed/                 # Sanitized text & intermediate metadata
│   └── vectordb/                  # Persistent ChromaDB vector data
├── docs/
│   ├── architecture.md            # Clean Architecture specification
│   ├── setup.md                   # Environment setup guide
│   └── api.md                     # REST API reference documentation
├── ingestion/
│   ├── __init__.py
│   ├── loader.py                  # Document loader for PDF/TXT/MD
│   ├── extractor.py               # Metadata extraction
│   ├── cleaner.py                 # Text sanitization & header removal
│   ├── section_parser.py          # Document section parser
│   ├── chunker.py                 # Overlapping text chunker
│   ├── embedder.py                # HuggingFace Sentence Transformers wrapper
│   └── ingest.py                  # Ingestion workflow orchestrator
├── llm/
│   ├── __init__.py
│   ├── gemini.py                  # Google Gemini API wrapper
│   ├── prompts.py                 # System grounding prompts & templates
│   └── response_formatter.py      # LLM output formatter
├── rag/
│   ├── __init__.py
│   ├── retriever.py               # Vector similarity search retriever
│   ├── reranker.py                # Cross-encoder passage reranker
│   ├── confidence.py             # Grounding & similarity confidence evaluator
│   ├── citations.py               # Citation extraction & source tracking
│   └── pipeline.py                # Core RAG pipeline orchestrator
├── vectorstore/
│   ├── __init__.py
│   ├── chroma_manager.py          # ChromaDB client lifecycle manager
│   └── collections.py             # Vector collection CRUD wrapper
├── tests/
│   ├── __init__.py
│   ├── test_loader.py             # Loader unit tests
│   ├── test_chunker.py            # Chunker unit tests
│   ├── test_retriever.py          # Retriever unit tests
│   └── test_api.py                # FastAPI endpoint integration tests
├── scripts/
│   ├── run_ingestion.py           # Standalone ingestion execution script
│   └── evaluate_rag.py            # RAG performance evaluation script
├── .gitignore
├── .env.example
├── LICENSE                        # MIT License
├── README.md                      # Project documentation
└── requirements.txt               # Dependency specifications
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.12+
- Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### 2. Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/ml-rag-agent.git
cd ml-rag-agent

# Create virtual environment
python -m venv venv

# Activate environment (Linux/macOS)
source venv/bin/activate
# Activate environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file from the provided template:
```bash
cp .env.example .env
```

Update your `.env` parameters:
```env
GEMINI_API_KEY="your-actual-gemini-api-key"
GEMINI_MODEL_NAME="gemini-1.5-flash"
EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
LOG_LEVEL="INFO"
```

---

## Building the Vector Database

Place your Machine Learning PDFs, Scikit-learn documentation, or KTU notes into `data/raw/`, then trigger ingestion:

### Via CLI:
```bash
python app/cli.py ingest --path data/raw
```

### Via Python Script:
```bash
python scripts/run_ingestion.py
```

---

## Running the FastAPI Application

Launch the server using `uvicorn`:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Interactive API Documentation:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Running Automated Tests

Run the test suite with `pytest`:
```bash
pytest -v
```

---

## Future Roadmap

- [ ] **Advanced Reranking**: Integrate full Cross-Encoder models (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- [ ] **Hybrid Search**: Combine dense vector embeddings with sparse BM25 keyword search.
- [ ] **n8n Automation**: Deploy live IMAP email trigger workflow for automated query response handling.
- [ ] **Streaming Responses**: Support Server-Sent Events (SSE) for streaming Gemini LLM output.
- [ ] **Multi-Modal Support**: Expand document loader to extract tables, diagrams, and figures.

---

## License

This project is licensed under the [MIT License](LICENSE).
