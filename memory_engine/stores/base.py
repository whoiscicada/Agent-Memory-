from abc import ABC, abstractmethod
from typing import Optional

from memory_engine.models.memory import MemoryType


class VectorSearchResult:
    __slots__ = ("memory_id", "score")

    def __init__(self, memory_id: str, score: float) -> None:
        self.memory_id = memory_id
        self.score = score


class VectorStore(ABC):
    @abstractmethod
    async def upsert(
        self,
        memory_id: str,
        vector: list[float],
        agent_id: str,
        memory_type: MemoryType,
        metadata: dict,
    ) -> None:
        ...

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        agent_id: str,
        top_k: int,
        memory_type: Optional[MemoryType] = None,
        include_global: bool = True,
    ) -> list[VectorSearchResult]:
        ...

    @abstractmethod
    async def delete(self, memory_id: str, agent_id: str) -> None:
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Create index/collection if not exists."""
        ...
