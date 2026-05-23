from typing import Optional

from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential

from memory_engine.config import settings
from memory_engine.models.memory import MemoryType
from memory_engine.stores.base import VectorSearchResult, VectorStore

GLOBAL_AGENT_ID = "__global__"


class PineconeStore(VectorStore):
    def __init__(self) -> None:
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name
        self._index = None

    async def initialize(self) -> None:
        existing = [i.name for i in self._pc.list_indexes()]
        if self._index_name not in existing:
            self._pc.create_index(
                name=self._index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = self._pc.Index(self._index_name)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def upsert(
        self,
        memory_id: str,
        vector: list[float],
        agent_id: str,
        memory_type: MemoryType,
        metadata: dict,
    ) -> None:
        assert self._index is not None, "Call initialize() first"
        self._index.upsert(
            vectors=[{
                "id": memory_id,
                "values": vector,
                "metadata": {"agent_id": agent_id, "memory_type": memory_type.value, **metadata},
            }],
            namespace=agent_id,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def search(
        self,
        vector: list[float],
        agent_id: str,
        top_k: int,
        memory_type: Optional[MemoryType] = None,
        include_global: bool = True,
    ) -> list[VectorSearchResult]:
        assert self._index is not None, "Call initialize() first"

        filter_expr: Optional[dict] = None
        if memory_type:
            filter_expr = {"memory_type": {"$eq": memory_type.value}}

        results = []
        namespaces = [agent_id]
        if include_global and agent_id != GLOBAL_AGENT_ID:
            namespaces.append(GLOBAL_AGENT_ID)

        for ns in namespaces:
            resp = self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=ns,
                filter=filter_expr,
                include_metadata=False,
            )
            for match in resp.matches:
                results.append(VectorSearchResult(memory_id=match.id, score=match.score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def delete(self, memory_id: str, agent_id: str) -> None:
        assert self._index is not None, "Call initialize() first"
        self._index.delete(ids=[memory_id], namespace=agent_id)
