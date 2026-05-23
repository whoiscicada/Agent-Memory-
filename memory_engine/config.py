from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Vector store
    vector_store: str = "weaviate"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "memory-engine"
    weaviate_url: str = "http://localhost:8080"

    # Embeddings
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"

    # Database
    database_url: str = "postgresql+asyncpg://memory:memory@localhost:5432/memory_engine"

    # Lifecycle
    episodic_ttl_days: int = 30
    semantic_ttl_days: int = 90
    summarization_threshold_days: int = 7
    max_memories_per_agent: int = 10000


settings = Settings()
