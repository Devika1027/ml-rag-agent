# Architecture Design Document - ML RAG Agent

## Overview

The **Production Machine Learning RAG Agent** is designed following **Clean Architecture principles**. The application isolates data access, vector storage, natural language processing, LLM generation, and web API handlers into distinct, decoupled modules.

---

## Clean Architecture Layers

```
+-----------------------------------------------------------------------+
|                             API Layer                                 |
|          (FastAPI, Routes, Pydantic Schemas, Dependency Injection)    |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                          RAG Pipeline Layer                           |
|        (Pipeline Orchestrator, Reranker, Confidence, Citations)       |
+-----------------------------------------------------------------------+
             |                                              |
             v                                              v
+---------------------------+                +--------------------------+
|     Ingestion Subsystem   |                |        LLM Subsystem     |
| (Loader, Cleaner, Chunker)|                |  (Google Gemini Client,  |
|       + VectorStore       |                |    System Prompts,       |
|    (ChromaDB Manager)     |                |   Response Formatter)    |
+---------------------------+                +--------------------------+
```

---

## Component Breakdown

### 1. Core & Config (`app/`)
- `config.py`: Manages settings dynamically via `pydantic-settings` and `.env` files. Employs `pathlib.Path` for cross-platform file paths.
- `cli.py`: Provides entrypoints for shell execution of ingestion and RAG querying.

### 2. Ingestion Subsystem (`ingestion/`)
- `loader.py`: Handles multi-format document reading (PDF, TXT, MD).
- `extractor.py`: Extracts document metadata (filename, page numbers, category).
- `cleaner.py`: Sanitizes text, normalizes whitespace, strips PDF header/footer artifacts.
- `section_parser.py`: Groups text into logical hierarchical document sections.
- `chunker.py`: Performs character/token overlapping text splitting.
- `embedder.py`: Wraps HuggingFace `all-MiniLM-L6-v2` SentenceTransformers.
- `ingest.py`: Orchestrates the complete end-to-end ingestion workflow.

### 3. Vector Database Layer (`vectorstore/`)
- `chroma_manager.py`: Connects to persistent ChromaDB database.
- `collections.py`: Enforces vector query, insert, and metadata filtering contracts.

### 4. RAG Engine Layer (`rag/`)
- `retriever.py`: Fetches top $K$ dense vector matches from ChromaDB.
- `reranker.py`: Cross-encoder reranking engine for score re-ordering.
- `confidence.py`: Computes grounding and similarity confidence scores.
- `citations.py`: Extracts and deduplicates source document references.
- `pipeline.py`: Unifies retrieval, prompt generation, LLM completion, and scoring.

### 5. LLM Layer (`llm/`)
- `gemini.py`: Client for Google Gemini API (`google-genai` SDK) with retries and safety settings.
- `prompts.py`: System instructions and prompt templates enforcing zero-hallucination grounding.
- `response_formatter.py`: Formats raw completions into structured client data.

### 6. API Layer (`api/`)
- `main.py`: FastAPI server setup with CORS, lifespan handlers, and error handlers.
- `routes.py`: Endpoint handlers (`/health`, `/query`, `/ingest`).
- `schemas.py`: Pydantic request/response payload validations.
- `dependencies.py`: Dependency injection providers for FastAPI.

---

## Data Flow for a RAG Query

1. **Client Request**: Client sends `POST /api/v1/query` with question JSON payload.
2. **Retrieval**: `VectorRetriever` embeds query via SentenceTransformers and queries ChromaDB for top $K$ context chunks.
3. **Reranking**: `CrossEncoderReranker` scores candidate chunks.
4. **Prompt Assembly**: `llm.prompts` formats context chunks into `RAG_PROMPT_TEMPLATE`.
5. **LLM Generation**: `GeminiLLMClient` sends grounded prompt to Google Gemini API.
6. **Scoring & Citations**: `ConfidenceEvaluator` calculates confidence score; `CitationTracker` formats source citations.
7. **Response**: FastAPI returns structured `RAGQueryResponse`.
