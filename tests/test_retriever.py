from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from memory_engine.core.retriever import MemoryRetriever
from memory_engine.core.writer import MemoryWriter
from memory_engine.models.memory import MemoryCreate, MemorySearch, MemoryType
from memory_engine.models.schemas import MemoryRow
from memory_engine.stores.base import VectorSearchResult


@pytest.mark.asyncio
async def test_search_returns_memories(db, mock_vector_store, mock_embedding_svc):
    # Write a memory first
    writer = MemoryWriter(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    mem = await writer.write(MemoryCreate(
        agent_id="retriever-agent",
        memory_type=MemoryType.SEMANTIC,
        content="Python sorting technique",
    ))

    mock_vector_store.search.return_value = [
        VectorSearchResult(memory_id=mem.id, score=0.9)
    ]

    retriever = MemoryRetriever(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    results = await retriever.search(MemorySearch(
        query="how to sort",
        agent_id="retriever-agent",
        top_k=5,
    ))

    assert len(results) == 1
    assert results[0].id == mem.id
    assert results[0].content == "Python sorting technique"


@pytest.mark.asyncio
async def test_search_increments_access_count(db, mock_vector_store, mock_embedding_svc):
    writer = MemoryWriter(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    mem = await writer.write(MemoryCreate(
        agent_id="count-agent",
        memory_type=MemoryType.EPISODIC,
        content="Accessed memory",
    ))

    mock_vector_store.search.return_value = [VectorSearchResult(memory_id=mem.id, score=0.8)]

    retriever = MemoryRetriever(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    results = await retriever.search(MemorySearch(query="test", agent_id="count-agent"))
    assert results[0].access_count == 1

    results = await retriever.search(MemorySearch(query="test", agent_id="count-agent"))
    assert results[0].access_count == 2


@pytest.mark.asyncio
async def test_search_skips_expired(db, mock_vector_store, mock_embedding_svc):
    from sqlalchemy import select
    writer = MemoryWriter(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    mem = await writer.write(MemoryCreate(
        agent_id="expiry-agent",
        memory_type=MemoryType.EPISODIC,
        content="Expired memory",
    ))

    # Manually expire the row
    from sqlalchemy import update
    from memory_engine.models.schemas import MemoryRow
    await db.execute(
        update(MemoryRow).where(MemoryRow.id == mem.id).values(expires_at=datetime.utcnow() - timedelta(days=1))
    )
    await db.commit()

    mock_vector_store.search.return_value = [VectorSearchResult(memory_id=mem.id, score=0.9)]
    retriever = MemoryRetriever(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    results = await retriever.search(MemorySearch(query="test", agent_id="expiry-agent"))
    assert results == []


@pytest.mark.asyncio
async def test_get_by_id(db, mock_vector_store, mock_embedding_svc):
    writer = MemoryWriter(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    mem = await writer.write(MemoryCreate(
        agent_id="byid-agent",
        memory_type=MemoryType.PROCEDURAL,
        content="Strategy content",
    ))

    retriever = MemoryRetriever(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    fetched = await retriever.get_by_id(mem.id)
    assert fetched is not None
    assert fetched.id == mem.id

    missing = await retriever.get_by_id("does-not-exist")
    assert missing is None
