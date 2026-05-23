from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from memory_engine.api.deps import get_db, get_retriever
from memory_engine.core.retriever import MemoryRetriever
from memory_engine.models.memory import Memory, MemorySearch

router = APIRouter(prefix="/memories", tags=["search"])


@router.post("/search", response_model=list[Memory])
async def search_memories(
    payload: MemorySearch,
    db: AsyncSession = Depends(get_db),
):
    retriever: MemoryRetriever = await get_retriever(db)
    return await retriever.search(payload)
