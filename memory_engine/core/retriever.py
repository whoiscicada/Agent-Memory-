from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from memory_engine.embeddings.base import EmbeddingService
from memory_engine.models.memory import Memory, MemorySearch, MemoryType
from memory_engine.models.schemas import MemoryAccessLog, MemoryRow
from memory_engine.stores.base import VectorStore


class MemoryRetriever:
    def __init__(self, db: AsyncSession, vector_store: VectorStore, embedding_svc: EmbeddingService) -> None:
        self._db = db
        self._vs = vector_store
        self._emb = embedding_svc

    async def search(self, payload: MemorySearch) -> list[Memory]:
        vector = await self._emb.embed(payload.query)
        hits = await self._vs.search(
            vector=vector,
            agent_id=payload.agent_id,
            top_k=payload.top_k,
            memory_type=payload.memory_type,
            include_global=payload.include_global,
        )
        if not hits:
            return []

        ids = [h.memory_id for h in hits]
        score_map = {h.memory_id: h.score for h in hits}

        result = await self._db.execute(
            select(MemoryRow).where(MemoryRow.id.in_(ids))
        )
        rows = result.scalars().all()

        now = datetime.utcnow()
        memories = []
        for row in rows:
            if row.expires_at and row.expires_at < now:
                continue
            row.access_count += 1
            row.last_accessed_at = now
            self._db.add(MemoryAccessLog(
                memory_id=row.id,
                accessed_at=now,
                query_snippet=payload.query[:512],
            ))
            mem = self._row_to_model(row)
            memories.append((score_map.get(row.id, 0.0), mem))

        await self._db.commit()

        memories.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in memories]

    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        result = await self._db.execute(
            select(MemoryRow).where(MemoryRow.id == memory_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._row_to_model(row)

    async def list_by_agent(self, agent_id: str, limit: int = 100) -> list[Memory]:
        result = await self._db.execute(
            select(MemoryRow)
            .where(MemoryRow.agent_id == agent_id)
            .order_by(MemoryRow.created_at.desc())
            .limit(limit)
        )
        return [self._row_to_model(r) for r in result.scalars().all()]

    @staticmethod
    def _row_to_model(row: MemoryRow) -> Memory:
        return Memory(
            id=row.id,
            agent_id=row.agent_id,
            memory_type=MemoryType(row.memory_type),
            content=row.content,
            metadata=row.metadata_,
            importance_score=row.importance_score,
            access_count=row.access_count,
            created_at=row.created_at,
            last_accessed_at=row.last_accessed_at,
            expires_at=row.expires_at,
        )
