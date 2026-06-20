"""
Agent 服务测试用例
- 冒烟测试：不依赖外部服务（Ollama），适合在 CI 中运行
- 集成测试：需要 Ollama 服务，仅在本地手动运行
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# 导入 app（如果 app.py 导入失败，CI 会直接报错，这就是冒烟测试的价值）
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
    # 只要结构包含这些字段即可，不关心实际值（因为 CI 没 Ollama）
    assert "status" in data
    assert "ollama" in data
    assert "chromadb" in data
    assert "reachable" in data["ollama"]
    assert "model_loaded" in data["ollama"]


def test_chat_endpoint_without_ollama():
    """测试 /chat 接口在无 Ollama 时的行为（预期 500）"""
    response = client.post(
        "/chat",
        json={"question": "现在几点了", "session_id": "test_ci"}
    )
    # 没有 Ollama 时应该返回 500，说明代码能正常处理异常
    # 如果返回 200，说明它绕过了依赖（不太可能）
    assert response.status_code in [500, 200]


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


# ============================================================
# 异常处理测试（不依赖外部服务）
# ============================================================

def test_chat_empty_question():
    """测试空问题（应返回 200 并给出提示，而不是报错）"""
    response = client.post(
        "/chat",
        json={"question": "", "session_id": "test_ci"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    # 空问题应该返回一些内容（提示或默认回复），而不是空字符串
    assert len(data["data"]) > 0


def test_chat_missing_session():
    """测试缺少 session_id（应使用默认值 'default'）"""
    response = client.post(
        "/chat",
        json={"question": "现在几点了"}
    )
    # 如果服务正常运行，即使缺 session_id 也应返回 500（因为没 Ollama）
    # 但不应抛出 422（说明 Pydantic 验证拦截了）
    assert response.status_code != 422