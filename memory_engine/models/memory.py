from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    memory_type: MemoryType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    embedding: Optional[list[float]] = None


class MemoryCreate(BaseModel):
    agent_id: str
    memory_type: MemoryType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance_boost: float = Field(default=0.0, ge=0.0, le=0.5)


class MemorySearch(BaseModel):
    query: str
    agent_id: str
    memory_type: Optional[MemoryType] = None
    top_k: int = Field(default=5, ge=1, le=100)
    include_global: bool = True
