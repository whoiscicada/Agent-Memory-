from typing import Optional

import weaviate
import weaviate.classes as wvc
from tenacity import retry, stop_after_attempt, wait_exponential

from memory_engine.config import settings
from memory_engine.models.memory import MemoryType
from memory_engine.stores.base import VectorSearchResult, VectorStore

COLLECTION_NAME = "Memory"
GLOBAL_AGENT_ID = "__global__"


class WeaviateStore(VectorStore):
    def __init__(self) -> None:
        self._url = settings.weaviate_url
        self._client: Optional[weaviate.WeaviateClient] = None

    async def initialize(self) -> None:
        self._client = weaviate.connect_to_local(
            host=self._url.replace("http://", "").split(":")[0],
            port=int(self._url.split(":")[-1]),
        )
        if not self._client.collections.exists(COLLECTION_NAME):
            self._client.collections.create(
                name=COLLECTION_NAME,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                properties=[
                    wvc.config.Property(name="memory_id", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="agent_id", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="memory_type", data_type=wvc.config.DataType.TEXT),
                ],
            )

    def _collection(self):
        assert self._client is not None, "Call initialize() first"
        return self._client.collections.get(COLLECTION_NAME)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def upsert(
        self,
        memory_id: str,
        vector: list[float],
        agent_id: str,
        memory_type: MemoryType,
        metadata: dict,
    ) -> None:
        col = self._collection()
        col.data.insert(
            properties={
                "memory_id": memory_id,
                "agent_id": agent_id,
                "memory_type": memory_type.value,
            },
            vector=vector,
            uuid=memory_id,
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
        col = self._collection()

        filters = wvc.query.Filter.by_property("agent_id").equal(agent_id)
        if include_global and agent_id != GLOBAL_AGENT_ID:
            filters = filters | wvc.query.Filter.by_property("agent_id").equal(GLOBAL_AGENT_ID)
        if memory_type:
            filters = filters & wvc.query.Filter.by_property("memory_type").equal(memory_type.value)

        response = col.query.near_vector(
            near_vector=vector,
            limit=top_k,
            filters=filters,
            return_metadata=wvc.query.MetadataQuery(certainty=True),
            return_properties=["memory_id"],
        )

        results = []
        for obj in response.objects:
            score = obj.metadata.certainty or 0.0
            results.append(VectorSearchResult(memory_id=obj.properties["memory_id"], score=score))
        return results

    async def delete(self, memory_id: str, agent_id: str) -> None:
        col = self._collection()
        col.data.delete_by_id(memory_id)
