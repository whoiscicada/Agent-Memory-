from __future__ import annotations

from functools import lru_cache
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from memory_engine.config import settings
from memory_engine.core.lifecycle import LifecycleManager
from memory_engine.core.retriever import MemoryRetriever
from memory_engine.core.writer import MemoryWriter
from memory_engine.embeddings.openai_embeddings import OpenAIEmbeddingService
from memory_engine.stores import get_vector_store
from memory_engine.stores.base import VectorStore

_engine = create_async_engine(settings.database_url, pool_pre_ping=True)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


@lru_cache(maxsize=1)
def _get_vector_store() -> VectorStore:
    return get_vector_store()


@lru_cache(maxsize=1)
def _get_embedding_svc() -> OpenAIEmbeddingService:
    return OpenAIEmbeddingService()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with _session_factory() as session:
        yield session


async def get_writer(db: AsyncSession) -> MemoryWriter:
    return MemoryWriter(db=db, vector_store=_get_vector_store(), embedding_svc=_get_embedding_svc())


async def get_retriever(db: AsyncSession) -> MemoryRetriever:
    return MemoryRetriever(db=db, vector_store=_get_vector_store(), embedding_svc=_get_embedding_svc())


async def get_lifecycle(db: AsyncSession) -> LifecycleManager:
    return LifecycleManager(db=db, vector_store=_get_vector_store())
