"""
Agent 服务测试用例
- 冒烟测试：不依赖外部服务（Ollama），适合在 CI 中运行
- 集成测试：需要 Ollama 服务，仅在本地手动运行
"""
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# ============================================================
# 冒烟测试（不依赖外部服务，CI 可运行）
# ============================================================

def test_app_imports():
    """冒烟测试：确保 app 模块可以正常导入"""
    from app import app
    assert app is not None


def test_health_endpoint_structure():
    """测试 /health 接口返回结构正确（不验证实际值）"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["ok", "degraded"]   # 允许 CI 环境下为 degraded
    assert "ollama" in data
    assert "chromadb" in data
    assert "reachable" in data["ollama"]
    assert "model_loaded" in data["ollama"]


def test_chat_endpoint_without_ollama():
    """测试 /chat 接口在无 Ollama 时的行为（预期 500）"""
    # 检查 Ollama 是否可达，如果可达则跳过此测试
    try:
        import requests
        resp = requests.get("http://localhost:11434", timeout=1)
    except:
        pass
    else:
        if resp.status_code == 200:
            pytest.skip("Ollama 正在运行，此测试仅适用于无 Ollama 环境")
   
    response = client.post(
        "/chat",
        json={"question": "现在几点了", "session_id": "test_ci"}
    )
    assert response.status_code == 500


# ============================================================
# 需要 Ollama 的测试（跳过，仅在本地运行）
# ============================================================

@pytest.mark.skip(reason="需要 Ollama 服务，仅在本地运行")
def test_chat_time():
    """测试时间查询（需要 Ollama）"""
    response = client.post(
        "/chat",
        json={"question": "现在几点了", "session_id": "test_local"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert "时间" in data["data"] or "点" in data["data"]


@pytest.mark.skip(reason="需要 Ollama 服务，仅在本地运行")
def test_chat_math():
    """测试数学计算（需要 Ollama）"""
    response = client.post(
        "/chat",
        json={"question": "1+1等于几", "session_id": "test_local"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert "2" in data["data"]


@pytest.mark.skip(reason="需要 Ollama 服务，仅在本地运行")
def test_chat_rag():
    """测试 RAG 检索（需要 Ollama + Chroma）"""
    response = client.post(
        "/chat",
        json={"question": "欧阳超擅长什么", "session_id": "test_local"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert len(data["data"]) > 0


@pytest.mark.skip(reason="需要 Ollama 服务，仅在本地运行")
def test_chat_empty_question():
    """测试空问题（需要 Ollama）"""
    response = client.post(
        "/chat",
        json={"question": "", "session_id": "test_ci"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert len(data["data"]) > 0


def test_chat_missing_session():
    """测试缺少 session_id（应使用默认值 'default'）"""
    response = client.post(
        "/chat",
        json={"question": "现在几点了"}
    )
    # 如果没有 Ollama，返回 500；但不应抛出 422（验证通过）
    assert response.status_code != 422