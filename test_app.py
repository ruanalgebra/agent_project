"""
Agent Service Test Suite

Test cases are divided into two layers:
- Smoke tests: Run in CI, no external dependencies (Ollama not required).
- Integration tests: Require Ollama service, run only locally.
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# ============================================================
# Smoke Tests (CI-ready, no external dependencies)
# ============================================================

def test_app_imports():
    """Smoke test: ensure app module can be imported."""
    from app import app
    assert app is not None


def test_health_endpoint_structure():
    """Check /health returns expected structure (values may vary in CI)."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["ok", "degraded"]   # CI may report degraded
    assert "ollama" in data
    assert "chromadb" in data
    assert "reachable" in data["ollama"]
    assert "model_loaded" in data["ollama"]


def test_chat_endpoint_without_ollama():
    """
    Test /chat behavior when Ollama is unavailable (expected 500).
    Skipped if Ollama is reachable (local development).
    """
    # Check if Ollama is reachable; skip if so
    try:
        import requests
        resp = requests.get("http://localhost:11434", timeout=1)
    except:
        pass
    else:
        if resp.status_code == 200:
            pytest.skip("Ollama is running; this test only applies to no-Ollama environments")

    response = client.post(
        "/chat",
        json={"question": "现在几点了", "session_id": "test_ci"}
    )
    assert response.status_code == 500


# ============================================================
# Integration Tests (require Ollama, skipped by default in CI)
# ============================================================

@pytest.mark.skip(reason="Requires Ollama service, run locally only")
def test_chat_time():
    """Test time query (requires Ollama)."""
    response = client.post(
        "/chat",
        json={"question": "现在几点了", "session_id": "test_local"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert "时间" in data["data"] or "点" in data["data"]


@pytest.mark.skip(reason="Requires Ollama service, run locally only")
def test_chat_math():
    """Test math calculation (requires Ollama)."""
    response = client.post(
        "/chat",
        json={"question": "1+1等于几", "session_id": "test_local"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert "2" in data["data"]


@pytest.mark.skip(reason="Requires Ollama service, run locally only")
def test_chat_rag():
    """Test RAG retrieval (requires Ollama + Chroma)."""
    response = client.post(
        "/chat",
        json={"question": "欧阳超擅长什么", "session_id": "test_local"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert len(data["data"]) > 0


@pytest.mark.skip(reason="Requires Ollama service, run locally only")
def test_chat_empty_question():
    """Test empty question handling (requires Ollama)."""
    response = client.post(
        "/chat",
        json={"question": "", "session_id": "test_ci"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert len(data["data"]) > 0


def test_chat_missing_session():
    """Test missing session_id; should default to 'default'."""
    response = client.post(
        "/chat",
        json={"question": "现在几点了"}
    )
    # Without Ollama, returns 500; but should not raise 422 (validation passed)
    assert response.status_code != 422