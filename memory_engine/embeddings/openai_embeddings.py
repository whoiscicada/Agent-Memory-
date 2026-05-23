from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from memory_engine.config import settings
from memory_engine.embeddings.base import EmbeddingService


class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            input=[text],
            model=self._model,
        )
        return response.data[0].embedding

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
