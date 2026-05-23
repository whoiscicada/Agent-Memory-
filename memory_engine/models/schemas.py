from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MemoryRow(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    memory_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default={})
    importance_score: Mapped[float] = mapped_column(Float, default=0.5)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    access_logs: Mapped[list[MemoryAccessLog]] = relationship(
        "MemoryAccessLog", back_populates="memory", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_memories_agent_type", "agent_id", "memory_type"),
        Index("ix_memories_expires_at", "expires_at"),
    )


class MemoryAccessLog(Base):
    __tablename__ = "memory_access_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    accessed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    query_snippet: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    memory: Mapped[MemoryRow] = relationship("MemoryRow", back_populates="access_logs")


class SummarizationJob(Base):
    __tablename__ = "summarization_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    memories_summarized: Mapped[int] = mapped_column(Integer, default=0)
    summary_memory_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
