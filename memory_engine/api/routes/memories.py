from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from memory_engine.api.deps import get_db, get_retriever, get_writer
from memory_engine.core.retriever import MemoryRetriever
from memory_engine.core.writer import MemoryWriter
from memory_engine.models.memory import Memory, MemoryCreate

router = APIRouter(prefix="/memories", tags=["memories"])


@router.post("", response_model=Memory, status_code=201)
async def create_memory(
    payload: MemoryCreate,
    db: AsyncSession = Depends(get_db),
):
    writer: MemoryWriter = await get_writer(db)
    return await writer.write(payload)


@router.get("/{memory_id}", response_model=Memory)
async def get_memory(
    memory_id: str,
    db: AsyncSession = Depends(get_db),
):
    retriever: MemoryRetriever = await get_retriever(db)
    mem = await retriever.get_by_id(memory_id)
    if mem is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return mem


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: str,
    db: AsyncSession = Depends(get_db),
):
    from memory_engine.api.deps import _get_vector_store
    from memory_engine.models.schemas import MemoryRow
    from sqlalchemy import select

    result = await db.execute(select(MemoryRow).where(MemoryRow.id == memory_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    vs = _get_vector_store()
    await vs.delete(memory_id, row.agent_id)
    await db.delete(row)
    await db.commit()


@router.get("/agent/{agent_id}", response_model=list[Memory])
async def list_agent_memories(
    agent_id: str,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    retriever: MemoryRetriever = await get_retriever(db)
    return await retriever.list_by_agent(agent_id, limit=limit)
