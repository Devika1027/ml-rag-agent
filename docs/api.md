# API Endpoint Documentation - ML RAG Agent

Base URL: `http://localhost:8000/api/v1`

---

## 1. Health Check Endpoint

### `GET /health`
Verifies server health and ChromaDB connectivity status.

#### Response `200 OK`
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "vectordb_status": "connected"
}
```

---

## 2. RAG Query Endpoint

### `POST /query`
Executes full RAG workflow for Machine Learning questions.

#### Request Body
```json
{
  "query": "Explain how Support Vector Machines compute optimal hyperplanes.",
  "top_k": 5,
  "category_filter": "ml-book"
}
```

#### Response `200 OK`
```json
{
  "query": "Explain how Support Vector Machines compute optimal hyperplanes.",
  "answer": "Support Vector Machines (SVM) construct a decision boundary that maximizes the margin between different classes...",
  "confidence_score": 0.92,
  "sources": [
    {
      "filename": "svm_chapter.pdf",
      "page_number": 14,
      "category": "ml-book",
      "snippet": "The optimal margin hyperplane minimizes the norm of the weight vector subject to margin constraints...",
      "similarity_score": 0.89
    }
  ],
  "metadata": {
    "model": "gemini-1.5-flash",
    "retrieved_chunks_count": 5
  }
}
```

---

## 3. Ingestion Trigger Endpoint

### `POST /ingest`
Triggers reading, splitting, embedding, and vector storage of documents.

#### Request Body
```json
{
  "target_path": "data/raw"
}
```

#### Response `200 OK`
```json
{
  "status": "success",
  "files_processed": 3,
  "total_chunks": 48,
  "details": {
    "directory": "data/raw"
  }
}
```
