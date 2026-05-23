from __future__ import annotations

import structlog
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from memory_engine.api.deps import _engine, get_db, get_lifecycle
from memory_engine.api.routes import memories, search
from memory_engine.core.lifecycle import LifecycleManager
from memory_engine.models.schemas import Base
from memory_engine.stores import get_vector_store

log = structlog.get_logger()

app = FastAPI(
    title="Memory Engine",
    description="Persistent long-term memory layer for AI agents",
    version="0.1.0",
)

app.include_router(memories.router)
app.include_router(search.router)


@app.on_event("startup")
async def startup() -> None:
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    vs = get_vector_store()
    await vs.initialize()
    log.info("memory_engine.started")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/admin/lifecycle/run", tags=["admin"])
async def run_lifecycle(db: AsyncSession = Depends(get_db)):
    manager: LifecycleManager = await get_lifecycle(db)
    result = await manager.run()
    return result
