from __future__ import annotations

from src.agent.context import AgentContext
from src.agent.loop import AgentLoop
from src.agent.memory import InMemoryMemoryStore
from src.agent.memory_integration import MemoryIntegration
from src.agent.task import Task


def make_task() -> Task:
    return Task(
        description="remember the execution result",
    )


def test_agent_loop_can_use_memory() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    loop = AgentLoop(
        memory=memory,
    )

    task = make_task()

    result = loop.run(task)

    assert result.task_id == task.id


def test_memory_is_scoped_to_task() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    task = make_task()

    entry = memory.store_for_task(
        task.id,
        "previous execution result",
    )

    memories = memory.retrieve_for_task(task.id)

    assert memories == (entry,)


def test_memory_does_not_leak_between_tasks() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    first_task = make_task()
    second_task = make_task()

    memory.store_for_task(
        first_task.id,
        "first task memory",
    )

    memories = memory.retrieve_for_task(
        second_task.id,
    )

    assert memories == ()


def test_memory_is_scoped_to_agent() -> None:
    store = InMemoryMemoryStore()

    first_agent = MemoryIntegration(
        store,
        agent_id="agent-a",
    )

    second_agent = MemoryIntegration(
        store,
        agent_id="agent-b",
    )

    task = make_task()

    first_agent.store_for_task(
        task.id,
        "agent A memory",
    )

    assert first_agent.retrieve_for_task(
        task.id,
    )

    assert second_agent.retrieve_for_task(
        task.id,
    ) == ()


def test_memory_enriches_context() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    task = make_task()

    memory.store_for_task(
        task.id,
        "remember this",
    )

    context = AgentContext(
        task=task,
        state="idle",
    )

    enriched = memory.enrich_context(context)

    assert len(enriched.memories) == 1
    assert enriched.memories[0].content == "remember this"


def test_memory_enrichment_preserves_context_identity() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    task = make_task()

    context = AgentContext(
        task=task,
        state="idle",
    )

    enriched = memory.enrich_context(context)

    assert enriched is not context
    assert enriched.task_id == context.task_id
    assert enriched.task is context.task


def test_execution_output_can_be_persisted() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    task = make_task()

    context = AgentContext(
        task=task,
        state="running",
    )

    entry = memory.store_execution_output(
        context,
        "execution completed successfully",
    )

    assert entry is not None
    assert entry.content == "execution completed successfully"
    assert entry.task_id == task.id
    assert entry.agent_id == "test-agent"