from memory_engine.config import settings
from memory_engine.stores.base import VectorSearchResult, VectorStore


def get_vector_store() -> VectorStore:
    if settings.vector_store == "pinecone":
        from memory_engine.stores.pinecone_store import PineconeStore
        return PineconeStore()
    else:
        from memory_engine.stores.weaviate_store import WeaviateStore
        return WeaviateStore()


__all__ = ["VectorStore", "VectorSearchResult", "get_vector_store"]
