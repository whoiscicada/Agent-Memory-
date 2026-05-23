# Agent Memory Engine

Persistent long-term memory layer for AI agents. Stores, indexes, and retrieves context across sessions — making agents genuinely stateful.

## Features

- **Three memory types**: Episodic (what happened), Semantic (facts learned), Procedural (how to do X)
- **Vector search**: OpenAI embeddings + Pinecone or Weaviate (pluggable)
- **Scoped access**: per-agent namespacing, shared global pool
- **Lifecycle management**: TTL eviction, importance decay, auto-summarization via LLM
- **Drop-in adapter**: integrate any existing agent in 3 lines
- **FastAPI**: full REST API with auto-generated docs

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Fill in OPENAI_API_KEY (and PINECONE_API_KEY if using Pinecone)

# 2. Start the stack
docker compose up

# 3. API docs
open http://localhost:8000/docs
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI Layer                     │
│  POST /memories  GET /memories/search  DELETE ...   │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│               Memory Engine Core                     │
│  MemoryWriter │ MemoryRetriever │ LifecycleManager  │
└──────┬─────────────────┬──────────────────┬─────────┘
       │                 │                  │
┌──────▼──────┐  ┌───────▼──────┐  ┌───────▼──────────┐
│  Embedding  │  │ Vector Store │  │   PostgreSQL      │
│  (OpenAI)   │  │ (Pinecone /  │  │  metadata, TTL,   │
│             │  │  Weaviate)   │  │  scores, audit    │
└─────────────┘  └──────────────┘  └───────────────────┘
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/memories` | Write a new memory |
| GET | `/memories/{id}` | Fetch by ID |
| DELETE | `/memories/{id}` | Delete by ID |
| POST | `/memories/search` | Vector search |
| GET | `/memories/agent/{agent_id}` | List agent memories |
| POST | `/admin/lifecycle/run` | Trigger lifecycle pass |

## Adapter Usage

```python
from memory_engine.adapter import MemoryAdapter

adapter = MemoryAdapter(agent_id="agent-001", base_url="http://localhost:8000")

# Before task: inject context
context = await adapter.get_context(task_prompt)

# After task: persist outcome
await adapter.save(
    content=f"Task: {task_prompt}\nOutcome: {result}",
    memory_type="episodic",
    metadata={"success": True}
)
```

## Memory Types

| Type | TTL (default) | Use case |
|------|--------------|----------|
| Episodic | 30 days | Interaction logs, task outcomes |
| Semantic | 90 days | User preferences, extracted facts |
| Procedural | No expiry | Successful strategies, workflows |

## Configuration

```env
VECTOR_STORE=weaviate          # or "pinecone"
OPENAI_API_KEY=...
EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=postgresql+asyncpg://...
EPISODIC_TTL_DAYS=30
SEMANTIC_TTL_DAYS=90
SUMMARIZATION_THRESHOLD_DAYS=7
MAX_MEMORIES_PER_AGENT=10000
```

## Demo

Runs an agent across 5 sessions demonstrating memory reuse:

```bash
python demo/multi_session_demo.py
```

| Session | Demonstrates |
|---------|-------------|
| 1 | Solves problem → stores episodic memory |
| 2 | Recalls session 1 → no re-solving |
| 3 | Learns user preference → stores semantic memory |
| 4 | Applies preference without being told |
| 5 | Reuses successful strategy (procedural memory) |

## Development

```bash
# Install with dev deps
pip install -e ".[dev]"

# Run tests
pytest --cov=memory_engine tests/

# Run benchmarks
python benchmarks/retrieval_bench.py --corpus 1000 --queries 50
```

## Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| API | FastAPI |
| Vector store | Pinecone / Weaviate |
| Relational store | PostgreSQL (SQLAlchemy 2.x async) |
| Embeddings | OpenAI text-embedding-3-small |
| Containers | Docker + Docker Compose |
