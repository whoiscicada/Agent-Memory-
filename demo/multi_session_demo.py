"""
5-session demo: agent remembers across sessions using MemoryAdapter.

Run with:
    python demo/multi_session_demo.py

Requires the API server running at http://localhost:8000.
"""
from __future__ import annotations

import asyncio
import textwrap

from openai import AsyncOpenAI

from memory_engine.adapter.agent_adapter import MemoryAdapter
from memory_engine.models.memory import MemoryType

AGENT_ID = "demo-agent-001"
BASE_URL = "http://localhost:8000"

oai = AsyncOpenAI()


def banner(session: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  SESSION {session}: {title}")
    print(f"{'='*60}")


def print_memories(memories_block: str) -> None:
    if memories_block:
        print("\n[Retrieved memories]")
        print(memories_block)
    else:
        print("\n[No relevant memories found]")


async def agent_respond(task: str, context: str) -> str:
    prompt = context + f"User task: {task}"
    resp = await oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI agent. Use the memory context provided to give consistent, personalized responses."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
    )
    return resp.choices[0].message.content or ""


async def session_1(adapter: MemoryAdapter) -> None:
    banner(1, "Solving a problem — storing episodic memory")
    task = "Sort a large list of customer records by last name, then first name."

    context = await adapter.get_context(task)
    print_memories(context)

    response = await agent_respond(task, context)
    print(f"\n[Agent response]\n{response}")

    outcome = f"Task: {task}\nOutcome: Used Python's sorted() with a tuple key (last_name, first_name). Ran in O(n log n). User confirmed it worked correctly."
    mem_id = await adapter.save(
        content=outcome,
        memory_type=MemoryType.EPISODIC,
        metadata={"task_type": "sorting", "success": True},
        importance_boost=0.1,
    )
    print(f"\n[Stored episodic memory: {mem_id}]")


async def session_2(adapter: MemoryAdapter) -> None:
    banner(2, "Recalling session 1 — no re-solving from scratch")
    task = "I need to sort customer records again by last name."

    context = await adapter.get_context(task)
    print_memories(context)

    response = await agent_respond(task, context)
    print(f"\n[Agent response]\n{response}")
    print("\n[Agent recalled prior sorting strategy — did not re-derive from scratch]")


async def session_3(adapter: MemoryAdapter) -> None:
    banner(3, "Learning a user preference — storing semantic memory")
    task = "Can you help me write this function? I prefer functional style over loops."

    context = await adapter.get_context(task)
    print_memories(context)

    response = await agent_respond(task, context)
    print(f"\n[Agent response]\n{response}")

    preference = "User prefers functional programming style (map/filter/reduce, list comprehensions) over imperative loops. Always use functional patterns when writing Python for this user."
    mem_id = await adapter.save(
        content=preference,
        memory_type=MemoryType.SEMANTIC,
        metadata={"category": "user_preference", "domain": "coding_style"},
        importance_boost=0.3,
    )
    print(f"\n[Stored semantic memory (preference): {mem_id}]")


async def session_4(adapter: MemoryAdapter) -> None:
    banner(4, "Applying preference from session 3 — not told again")
    task = "Write a function to filter out inactive users from a list."

    context = await adapter.get_context(task)
    print_memories(context)

    response = await agent_respond(task, context)
    print(f"\n[Agent response]\n{response}")
    print("\n[Agent used functional style (filter/comprehension) without being reminded]")


async def session_5(adapter: MemoryAdapter) -> None:
    banner(5, "Reusing successful strategy — procedural memory")
    task = "Write a reusable data processing pipeline for transforming records."

    # First store a procedural memory from the patterns we've seen
    strategy = textwrap.dedent("""\
        Reusable data processing pattern for this user:
        1. Use functional composition (pipe functions together)
        2. Each step: pure function taking list → returning list
        3. Sort step: sorted(data, key=lambda r: (r.last_name, r.first_name))
        4. Filter step: filter(predicate_fn, data) or list comprehension
        5. Transform step: map(transform_fn, data)
        This pattern has been validated across multiple tasks for this user.
    """)
    proc_id = await adapter.save(
        content=strategy,
        memory_type=MemoryType.PROCEDURAL,
        metadata={"domain": "data_processing", "validated": True},
        importance_boost=0.4,
    )
    print(f"[Pre-stored procedural memory: {proc_id}]")

    context = await adapter.get_context(task)
    print_memories(context)

    response = await agent_respond(task, context)
    print(f"\n[Agent response]\n{response}")
    print("\n[Agent reused validated pipeline strategy from procedural memory]")


async def main() -> None:
    print("Memory Engine — 5-Session Demo")
    print(f"Agent ID: {AGENT_ID}")
    print(f"API: {BASE_URL}")

    async with MemoryAdapter(agent_id=AGENT_ID, base_url=BASE_URL) as adapter:
        await session_1(adapter)
        await asyncio.sleep(1)

        await session_2(adapter)
        await asyncio.sleep(1)

        await session_3(adapter)
        await asyncio.sleep(1)

        await session_4(adapter)
        await asyncio.sleep(1)

        await session_5(adapter)

    print(f"\n{'='*60}")
    print("Demo complete. All 5 sessions ran with memory reuse.")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
