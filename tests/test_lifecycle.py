from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import update

from memory_engine.core.lifecycle import LifecycleManager
from memory_engine.core.writer import MemoryWriter
from memory_engine.models.memory import MemoryCreate, MemoryType
from memory_engine.models.schemas import MemoryRow


@pytest.mark.asyncio
async def test_evict_expired(db, mock_vector_store, mock_embedding_svc):
    writer = MemoryWriter(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    mem = await writer.write(MemoryCreate(
        agent_id="lifecycle-agent",
        memory_type=MemoryType.EPISODIC,
        content="Will be evicted",
    ))

    await db.execute(
        update(MemoryRow).where(MemoryRow.id == mem.id).values(
            expires_at=datetime.utcnow() - timedelta(days=1)
        )
    )
    await db.commit()

    lm = LifecycleManager(db=db, vector_store=mock_vector_store)
    evicted = await lm._evict_expired()
    assert evicted >= 1
    mock_vector_store.delete.assert_called()


@pytest.mark.asyncio
async def test_decay_scores(db, mock_vector_store, mock_embedding_svc):
    writer = MemoryWriter(db=db, vector_store=mock_vector_store, embedding_svc=mock_embedding_svc)
    mem = await writer.write(MemoryCreate(
        agent_id="decay-agent",
        memory_type=MemoryType.SEMANTIC,
        content="Old memory",
    ))

    await db.execute(
        update(MemoryRow).where(MemoryRow.id == mem.id).values(
            last_accessed_at=datetime.utcnow() - timedelta(days=14),
            importance_score=0.8,
        )
    )
    await db.commit()

    lm = LifecycleManager(db=db, vector_store=mock_vector_store)
    decayed = await lm._decay_scores()
    assert decayed >= 1

    from sqlalchemy import select
    result = await db.execute(select(MemoryRow).where(MemoryRow.id == mem.id))
    row = result.scalar_one_or_none()
    if row:
        assert row.importance_score < 0.8
