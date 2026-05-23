from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from memory_engine.config import settings
from memory_engine.embeddings.base import EmbeddingService
from memory_engine.models.memory import Memory, MemoryCreate, MemoryType
from memory_engine.models.schemas import MemoryRow
from memory_engine.stores.base import VectorStore

_TTL_MAP = {
    MemoryType.EPISODIC: "episodic_ttl_days",
    MemoryType.SEMANTIC: "semantic_ttl_days",
    MemoryType.PROCEDURAL: None,
}


class MemoryWriter:
    def __init__(self, db: AsyncSession, vector_store: VectorStore, embedding_svc: EmbeddingService) -> None:
        self._db = db
        self._vs = vector_store
        self._emb = embedding_svc

    async def write(self, payload: MemoryCreate) -> Memory:
        await self._enforce_cap(payload.agent_id)

        vector = await self._emb.embed(payload.content)
        importance = min(0.5 + payload.importance_boost, 1.0)

        if payload.metadata.get("importance"):
            importance = min(importance + 0.2, 1.0)

        expires_at = self._compute_expiry(payload.memory_type)

        mem = Memory(
            agent_id=payload.agent_id,
            memory_type=payload.memory_type,
            content=payload.content,
            metadata=payload.metadata,
            importance_score=importance,
            expires_at=expires_at,
        )

        row = MemoryRow(
            id=mem.id,
            agent_id=mem.agent_id,
            memory_type=mem.memory_type.value,
            content=mem.content,
            metadata_=mem.metadata,
            importance_score=mem.importance_score,
            created_at=mem.created_at,
            last_accessed_at=mem.last_accessed_at,
            expires_at=mem.expires_at,
        )
        self._db.add(row)
        await self._db.flush()

        await self._vs.upsert(
            memory_id=mem.id,
            vector=vector,
            agent_id=mem.agent_id,
            memory_type=mem.memory_type,
            metadata=mem.metadata,
        )

        await self._db.commit()
        mem.embedding = vector
        return mem

    def _compute_expiry(self, memory_type: MemoryType) -> Optional[datetime]:
        attr = _TTL_MAP[memory_type]
        if attr is None:
            return None
        days = getattr(settings, attr)
        return datetime.utcnow() + timedelta(days=days)

    async def _enforce_cap(self, agent_id: str) -> None:
        result = await self._db.execute(
            select(MemoryRow)
            .where(MemoryRow.agent_id == agent_id)
            .order_by(MemoryRow.importance_score.asc())
            .limit(1)
        )
        count_result = await self._db.execute(
            select(MemoryRow.id).where(MemoryRow.agent_id == agent_id)
        )
        count = len(count_result.all())
        if count >= settings.max_memories_per_agent:
            oldest_low = result.scalar_one_or_none()
            if oldest_low:
                await self._db.delete(oldest_low)
                await self._vs.delete(oldest_low.id, agent_id)
