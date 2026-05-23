from __future__ import annotations

from typing import Any, Optional

import httpx

from memory_engine.models.memory import MemoryType


class MemoryAdapter:
    """Drop-in wrapper that gives any agent persistent memory in 3 lines."""

    def __init__(self, agent_id: str, base_url: str = "http://localhost:8000") -> None:
        self._agent_id = agent_id
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base, timeout=30.0)

    async def get_context(
        self,
        task_prompt: str,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
    ) -> str:
        """Return a formatted string of relevant memories to prepend to any prompt."""
        payload: dict[str, Any] = {
            "query": task_prompt,
            "agent_id": self._agent_id,
            "top_k": top_k,
            "include_global": True,
        }
        if memory_type:
            payload["memory_type"] = memory_type.value

        resp = await self._client.post("/memories/search", json=payload)
        resp.raise_for_status()
        memories = resp.json()

        if not memories:
            return ""

        lines = ["[MEMORY CONTEXT]"]
        for i, mem in enumerate(memories, 1):
            mtype = mem["memory_type"].upper()
            lines.append(f"{i}. [{mtype}] {mem['content']}")
        lines.append("[END MEMORY CONTEXT]\n")
        return "\n".join(lines)

    async def save(
        self,
        content: str,
        memory_type: MemoryType | str = MemoryType.EPISODIC,
        metadata: Optional[dict] = None,
        importance_boost: float = 0.0,
    ) -> str:
        """Persist a memory. Returns the memory ID."""
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        payload = {
            "agent_id": self._agent_id,
            "memory_type": memory_type.value,
            "content": content,
            "metadata": metadata or {},
            "importance_boost": importance_boost,
        }
        resp = await self._client.post("/memories", json=payload)
        resp.raise_for_status()
        return resp.json()["id"]

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> MemoryAdapter:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
