
from __future__ import annotations

from src.agent.context import AgentContext
from src.agent.memory import InMemoryMemoryStore
from src.agent.memory_integration import MemoryIntegration
from src.agent.task import Task


def make_task(
    description: str,
) -> Task:
    return Task(
        description=description,
    )


def test_retrieve_relevant_finds_matching_previous_task_memory() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    previous_task = make_task(
        "calculate the result of a mathematical expression",
    )

    current_task = make_task(
        "calculate the result of another mathematical expression",
    )

    entry = memory.store_for_task(
        previous_task.id,
        "The mathematical expression result was calculated successfully.",
    )

    results = memory.retrieve_relevant(current_task)

    assert results == (entry,)


def test_retrieve_relevant_ignores_unrelated_memories() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    previous_task = make_task(
        "calculate a mathematical expression",
    )

    current_task = make_task(
        "inspect a filesystem directory",
    )

    entry = memory.store_for_task(
        previous_task.id,
        "The mathematical expression was calculated.",
    )

    results = memory.retrieve_relevant(current_task)

    assert results == ()
    assert entry not in results


def test_retrieve_relevant_does_not_include_current_task_memories() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    current_task = make_task(
        "calculate a mathematical expression",
    )

    entry = memory.store_for_task(
        current_task.id,
        "The mathematical expression was calculated.",
    )

    results = memory.retrieve_relevant(current_task)

    assert results == ()
    assert entry not in results


def test_retrieve_relevant_preserves_agent_isolation() -> None:
    store = InMemoryMemoryStore()

    agent_a = MemoryIntegration(
        store,
        agent_id="agent-a",
    )

    agent_b = MemoryIntegration(
        store,
        agent_id="agent-b",
    )

    previous_task = make_task(
        "calculate a mathematical expression",
    )

    current_task = make_task(
        "calculate another mathematical expression",
    )

    agent_a.store_for_task(
        previous_task.id,
        "The mathematical expression was calculated.",
    )

    results = agent_b.retrieve_relevant(current_task)

    assert results == ()


def test_retrieve_relevant_ranks_by_token_overlap() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    first_task = make_task(
        "calculate a result",
    )

    second_task = make_task(
        "calculate a result",
    )

    current_task = make_task(
        "calculate a mathematical result",
    )

    lower_match = memory.store_for_task(
        first_task.id,
        "calculate result",
    )

    higher_match = memory.store_for_task(
        second_task.id,
        "calculate mathematical result",
    )

    results = memory.retrieve_relevant(current_task)

    assert results == (
        higher_match,
        lower_match,
    )


def test_retrieve_relevant_uses_creation_order_as_tie_breaker() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    first_task = make_task(
        "calculate a result",
    )

    second_task = make_task(
        "calculate a result",
    )

    current_task = make_task(
        "calculate a result",
    )

    first = memory.store_for_task(
        first_task.id,
        "calculate result",
    )

    second = memory.store_for_task(
        second_task.id,
        "calculate result",
    )

    results = memory.retrieve_relevant(current_task)

    assert results == (
        first,
        second,
    )


def test_retrieve_relevant_supports_limit() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    first_task = make_task("calculate a result")
    second_task = make_task("calculate a result")
    third_task = make_task("calculate a result")

    current_task = make_task(
        "calculate a mathematical result",
    )

    first = memory.store_for_task(
        first_task.id,
        "calculate mathematical result",
    )

    second = memory.store_for_task(
        second_task.id,
        "calculate mathematical result",
    )

    memory.store_for_task(
        third_task.id,
        "calculate mathematical result",
    )

    results = memory.retrieve_relevant(
        current_task,
        limit=2,
    )

    assert results == (
        first,
        second,
    )


def test_retrieve_relevant_zero_limit_returns_empty() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    current_task = make_task(
        "calculate a result",
    )

    results = memory.retrieve_relevant(
        current_task,
        limit=0,
    )

    assert results == ()


def test_retrieve_relevant_rejects_negative_limit() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    current_task = make_task(
        "calculate a result",
    )

    try:
        memory.retrieve_relevant(
            current_task,
            limit=-1,
        )
    except ValueError as exc:
        assert str(exc) == "limit must not be negative"
    else:
        raise AssertionError(
            "expected ValueError for negative limit"
        )


def test_enrich_context_uses_relevant_historical_memory() -> None:
    store = InMemoryMemoryStore()

    memory = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    previous_task = make_task(
        "calculate a mathematical expression",
    )

    current_task = make_task(
        "calculate another mathematical expression",
    )

    memory.store_for_task(
        previous_task.id,
        "The mathematical expression was calculated successfully.",
    )

    context = AgentContext(
        task=current_task,
        agent_id="test-agent",
        state="idle",
    )

    enriched = memory.enrich_context(context)

    assert len(enriched.memories) == 1
    assert (
        enriched.memories[0].content
        == "The mathematical expression was calculated successfully."
    )


def test_retrieve_preserves_context_agent_isolation() -> None:
    store = InMemoryMemoryStore()

    first_agent = MemoryIntegration(
        store,
        agent_id="agent-a",
    )

    second_agent = MemoryIntegration(
        store,
        agent_id="agent-b",
    )

    previous_task = make_task(
        "calculate a mathematical expression",
    )

    current_task = make_task(
        "calculate another mathematical expression",
    )

    first_agent.store_for_task(
        previous_task.id,
        "The mathematical expression was calculated.",
    )

    context = AgentContext(
        task=current_task,
        agent_id="agent-b",
        state="idle",
    )

    enriched = second_agent.enrich_context(context)

    assert enriched.memories == ()
