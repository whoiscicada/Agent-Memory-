from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import structlog
from openai import AsyncOpenAI
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from memory_engine.config import settings
from memory_engine.models.memory import MemoryCreate, MemoryType
from memory_engine.models.schemas import MemoryRow, SummarizationJob
from memory_engine.stores.base import VectorStore

log = structlog.get_logger()


class LifecycleManager:
    def __init__(self, db: AsyncSession, vector_store: VectorStore) -> None:
        self._db = db
        self._vs = vector_store
        self._oai = AsyncOpenAI(api_key=settings.openai_api_key)

    async def run(self) -> dict:
        evicted = await self._evict_expired()
        decayed = await self._decay_scores()
        summarized = await self._summarize_old_episodic()
        return {"evicted": evicted, "decayed": decayed, "summarized": summarized}

    async def _evict_expired(self) -> int:
        now = datetime.utcnow()
        result = await self._db.execute(
            select(MemoryRow).where(
                MemoryRow.expires_at.isnot(None),
                MemoryRow.expires_at < now,
            )
        )
        rows = result.scalars().all()
        count = 0
        for row in rows:
            await self._vs.delete(row.id, row.agent_id)
            await self._db.delete(row)
            count += 1
        await self._db.commit()
        log.info("lifecycle.evicted", count=count)
        return count

    async def _decay_scores(self) -> int:
        cutoff = datetime.utcnow() - timedelta(days=7)
        result = await self._db.execute(
            select(MemoryRow).where(MemoryRow.last_accessed_at < cutoff)
        )
        rows = result.scalars().all()
        count = 0
        for row in rows:
            row.importance_score = max(0.0, row.importance_score * 0.9)
            count += 1
        await self._db.commit()
        log.info("lifecycle.decayed", count=count)
        return count

    async def _summarize_old_episodic(self) -> int:
        threshold = datetime.utcnow() - timedelta(days=settings.summarization_threshold_days)

        result = await self._db.execute(
            select(MemoryRow.agent_id)
            .where(
                MemoryRow.memory_type == MemoryType.EPISODIC.value,
                MemoryRow.created_at < threshold,
            )
            .distinct()
        )
        agent_ids = [r[0] for r in result.all()]

        total_summarized = 0
        for agent_id in agent_ids:
            total_summarized += await self._summarize_agent(agent_id, threshold)

        return total_summarized

    async def _summarize_agent(self, agent_id: str, threshold: datetime) -> int:
        job_check = await self._db.execute(
            select(SummarizationJob).where(
                SummarizationJob.agent_id == agent_id,
                SummarizationJob.status == "running",
            )
        )
        if job_check.scalar_one_or_none():
            return 0

        window_start = threshold - timedelta(days=settings.summarization_threshold_days)
        job = SummarizationJob(
            agent_id=agent_id,
            window_start=window_start,
            window_end=threshold,
            status="running",
        )
        self._db.add(job)
        await self._db.flush()

        result = await self._db.execute(
            select(MemoryRow).where(
                MemoryRow.agent_id == agent_id,
                MemoryRow.memory_type == MemoryType.EPISODIC.value,
                MemoryRow.created_at < threshold,
            ).limit(50)
        )
        rows = result.scalars().all()
        if not rows:
            await self._db.delete(job)
            await self._db.commit()
            return 0

        content_block = "\n---\n".join(r.content for r in rows)
        try:
            resp = await self._oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Summarize the following episodic memories into a concise semantic summary. Preserve key facts, outcomes, and patterns. Be specific."},
                    {"role": "user", "content": content_block},
                ],
                max_tokens=512,
            )
            summary_text = resp.choices[0].message.content or ""
        except Exception as exc:
            log.error("lifecycle.summarize_failed", agent_id=agent_id, error=str(exc))
            job.status = "failed"
            await self._db.commit()
            return 0

        from memory_engine.embeddings.openai_embeddings import OpenAIEmbeddingService
        from memory_engine.core.writer import MemoryWriter

        emb_svc = OpenAIEmbeddingService()
        writer = MemoryWriter(db=self._db, vector_store=self._vs, embedding_svc=emb_svc)
        summary_mem = await writer.write(MemoryCreate(
            agent_id=agent_id,
            memory_type=MemoryType.SEMANTIC,
            content=summary_text,
            metadata={"source": "auto_summarization", "episodes_count": len(rows)},
            importance_boost=0.2,
        ))

        for row in rows:
            await self._vs.delete(row.id, row.agent_id)
            await self._db.delete(row)

        job.status = "completed"
        job.memories_summarized = len(rows)
        job.summary_memory_id = summary_mem.id
        job.completed_at = datetime.utcnow()
        await self._db.commit()
        log.info("lifecycle.summarized", agent_id=agent_id, count=len(rows))
        return len(rows)
