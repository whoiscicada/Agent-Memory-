"""
Retrieval benchmark: latency (p50/p95/p99), throughput, relevance.

Usage:
    python benchmarks/retrieval_bench.py --corpus 1000 --queries 50
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Any

import httpx
from faker import Faker

BASE_URL = "http://localhost:8000"
AGENT_ID = "bench-agent"

fake = Faker()


async def seed_memories(client: httpx.AsyncClient, n: int) -> list[str]:
    print(f"Seeding {n} memories...")
    ids = []
    batch = 50
    for i in range(0, n, batch):
        tasks = []
        for _ in range(min(batch, n - i)):
            payload = {
                "agent_id": AGENT_ID,
                "memory_type": fake.random_element(["episodic", "semantic", "procedural"]),
                "content": fake.paragraph(nb_sentences=3),
                "metadata": {"tag": fake.word()},
                "importance_boost": 0.0,
            }
            tasks.append(client.post("/memories", json=payload))
        responses = await asyncio.gather(*tasks)
        for r in responses:
            r.raise_for_status()
            ids.append(r.json()["id"])
        print(f"  seeded {min(i + batch, n)}/{n}", end="\r")
    print()
    return ids


async def bench_search(client: httpx.AsyncClient, n_queries: int) -> list[float]:
    queries = [fake.sentence() for _ in range(n_queries)]
    latencies = []
    for q in queries:
        payload = {"query": q, "agent_id": AGENT_ID, "top_k": 5}
        t0 = time.perf_counter()
        r = await client.post("/memories/search", json=payload)
        elapsed = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        latencies.append(elapsed)
    return latencies


async def bench_write_throughput(client: httpx.AsyncClient, n: int = 100) -> float:
    payloads = [
        {
            "agent_id": AGENT_ID + "-tput",
            "memory_type": "episodic",
            "content": fake.paragraph(),
            "metadata": {},
            "importance_boost": 0.0,
        }
        for _ in range(n)
    ]
    t0 = time.perf_counter()
    await asyncio.gather(*[client.post("/memories", json=p) for p in payloads])
    elapsed = time.perf_counter() - t0
    return n / elapsed


def percentile(data: list[float], p: float) -> float:
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def print_results(label: str, latencies: list[float]) -> None:
    print(f"\n{label}")
    print(f"  count : {len(latencies)}")
    print(f"  p50   : {percentile(latencies, 50):.1f} ms")
    print(f"  p95   : {percentile(latencies, 95):.1f} ms")
    print(f"  p99   : {percentile(latencies, 99):.1f} ms")
    print(f"  mean  : {statistics.mean(latencies):.1f} ms")


async def main(corpus_size: int, n_queries: int) -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        await seed_memories(client, corpus_size)

        print(f"\nRunning {n_queries} search queries...")
        latencies = await bench_search(client, n_queries)
        print_results(f"Search latency ({corpus_size} memories in corpus)", latencies)

        print("\nRunning write throughput test (100 concurrent writes)...")
        tput = await bench_write_throughput(client, 100)
        print(f"  throughput: {tput:.1f} writes/sec")

    print("\nSave these results to benchmarks/RESULTS.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(main(args.corpus, args.queries))
