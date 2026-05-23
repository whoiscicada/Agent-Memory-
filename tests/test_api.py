from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from memory_engine.api.main import app
from memory_engine.models.memory import Memory, MemoryType

import datetime


def _make_memory(**kwargs) -> Memory:
    defaults = dict(
        id="test-id-123",
        agent_id="api-agent",
        memory_type=MemoryType.EPISODIC,
        content="Test content",
        metadata={},
        importance_score=0.5,
        access_count=0,
        created_at=datetime.datetime.utcnow(),
        last_accessed_at=datetime.datetime.utcnow(),
        expires_at=None,
    )
    defaults.update(kwargs)
    return Memory(**defaults)


@pytest.mark.asyncio
async def test_create_memory():
    mock_writer = AsyncMock()
    mock_writer.write.return_value = _make_memory()

    with patch("memory_engine.api.routes.memories.get_writer", return_value=mock_writer):
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post("/memories", json={
                "agent_id": "api-agent",
                "memory_type": "episodic",
                "content": "Test content",
                "metadata": {},
                "importance_boost": 0.0,
            })
    assert resp.status_code == 201
    assert resp.json()["id"] == "test-id-123"


@pytest.mark.asyncio
async def test_get_memory_not_found():
    mock_retriever = AsyncMock()
    mock_retriever.get_by_id.return_value = None

    with patch("memory_engine.api.routes.memories.get_retriever", return_value=mock_retriever):
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.get("/memories/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_search_memories():
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = [_make_memory()]

    with patch("memory_engine.api.routes.search.get_retriever", return_value=mock_retriever):
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post("/memories/search", json={
                "query": "test query",
                "agent_id": "api-agent",
                "top_k": 5,
            })
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
