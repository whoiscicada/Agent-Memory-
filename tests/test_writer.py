from __future__ import annotations

import pytest
from sqlalchemy import select

from memory_engine.core.writer import MemoryWriter
from memory_engine.models.memory import MemoryCreate, MemoryType
from memory_engine.models.schemas import MemoryRow


@pytest.mark.asyncio
async def test_write_creates_db_row(db, mock_vector_store, mock_embedding_svc):
    writer = MemoryWriter(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    payload = MemoryCreate(
        agent_id="agent-test",
        memory_type=MemoryType.EPISODIC,
        content="Test memory content",
        metadata={"key": "value"},
    )
    mem = await writer.write(payload)

    assert mem.id is not None
    assert mem.agent_id == "agent-test"
    assert mem.memory_type == MemoryType.EPISODIC
    assert mem.importance_score == pytest.approx(0.5)

    result = await db.execute(select(MemoryRow).where(MemoryRow.id == mem.id))
    row = result.scalar_one_or_none()
    assert row is not None
    assert row.content == "Test memory content"


@pytest.mark.asyncio
async def test_write_calls_vector_store(db, mock_vector_store, mock_embedding_svc):
    writer = MemoryWriter(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    payload = MemoryCreate(
        agent_id="agent-vec",
        memory_type=MemoryType.SEMANTIC,
        content="Semantic fact",
    )
    mem = await writer.write(payload)

    mock_vector_store.upsert.assert_called_once()
    call_kwargs = mock_vector_store.upsert.call_args
    assert call_kwargs.kwargs["memory_id"] == mem.id
    assert call_kwargs.kwargs["agent_id"] == "agent-vec"


@pytest.mark.asyncio
async def test_write_importance_boost(db, mock_vector_store, mock_embedding_svc):
    writer = MemoryWriter(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    payload = MemoryCreate(
        agent_id="agent-boost",
        memory_type=MemoryType.PROCEDURAL,
        content="Strategy",
        importance_boost=0.3,
    )
    mem = await writer.write(payload)
    assert mem.importance_score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_write_episodic_sets_expiry(db, mock_vector_store, mock_embedding_svc):
    writer = MemoryWriter(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    payload = MemoryCreate(
        agent_id="agent-exp",
        memory_type=MemoryType.EPISODIC,
        content="Will expire",
    )
    mem = await writer.write(payload)
    assert mem.expires_at is not None


@pytest.mark.asyncio
async def test_write_procedural_no_expiry(db, mock_vector_store, mock_embedding_svc):
    writer = MemoryWriter(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    payload = MemoryCreate(
        agent_id="agent-proc",
        memory_type=MemoryType.PROCEDURAL,
        content="No expiry",
    )
    mem = await writer.write(payload)
    assert mem.expires_at is None
