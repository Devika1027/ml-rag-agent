"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify /api/v1/health status endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vectordb_status" in data


def test_rag_query_endpoint():
    """Verify /api/v1/query RAG request endpoint."""
    payload = {
        "query": "What is linear regression?",
        "top_k": 3,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert "answer" in data
    assert "confidence_score" in data
    assert isinstance(data["sources"], list)
