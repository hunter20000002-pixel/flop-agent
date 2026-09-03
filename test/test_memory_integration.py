from datetime import datetime, timezone

import pytest

from src.agent.context import AgentContext
from src.agent.memory import InMemoryMemoryStore, MemoryEntry
from src.agent.memory_integration import MemoryIntegration
from src.agent.observation import TechnocoreObservation
from src.agent.task import Task
from src.client import Message


def make_context(
    *,
    agent_id: str | None = None,
    description: str = "Memory integration test",
) -> AgentContext:
    task = Task(description=description)

    return AgentContext(
        task=task,
        agent_id=agent_id,
        state="running",
    )


def make_observation(
    *,
    room: str = "lobby",
    since: int = 100,
    messages: tuple[Message, ...] | None = None,
) -> TechnocoreObservation:
    if messages is None:
        messages = (
            Message(
                seq=101,
                timestamp="2026-09-03T10:00:00Z",
                short_did="did:key:test1",
                text="FLOP Agent observation test.",
                raw_line="101 | test | FLOP Agent observation test.",
            ),
            Message(
                seq=102,
                timestamp="2026-09-03T10:01:00Z",
                short_did="did:key:test2",
                text="Technocore environment changed.",
                raw_line="102 | test | Technocore environment changed.",
            ),
        )

    return TechnocoreObservation(
        room=room,
        since=since,
        messages=messages,
        observed_at=datetime(
            2026,
            9,
            3,
            10,
            2,
            tzinfo=timezone.utc,
        ),
    )


def test_memory_integration_retrieves_task_memories():
    context = make_context()
    store = InMemoryMemoryStore()

    entry = MemoryEntry(
        content="Previous execution result",
        task_id=context.task.id,
    )

    store.store(entry)

    integration = MemoryIntegration(store)

    memories = integration.retrieve(context)

    assert memories == (entry,)


def test_memory_integration_enriches_context():
    context = make_context()
    store = InMemoryMemoryStore()

    entry = MemoryEntry(
        content="Relevant previous result",
        task_id=context.task.id,
    )

    store.store(entry)

    integration = MemoryIntegration(store)

    enriched = integration.enrich_context(context)

    assert enriched is not context
    assert enriched.task.id == context.task.id
    assert enriched.memories == (entry,)


def test_memory_integration_preserves_context_identity():
    context = make_context(agent_id="agent-1")
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)

    enriched = integration.enrich_context(context)

    assert enriched.task.id == context.task.id
    assert enriched.agent_id == context.agent_id


def test_memory_integration_stores_memory_for_task():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)

    entry = integration.remember(
        context,
        "The agent completed the previous execution.",
    )

    assert entry.content == (
        "The agent completed the previous execution."
    )
    assert entry.task_id == context.task.id

    memories = store.query(task_id=context.task.id)

    assert memories == (entry,)


def test_memory_integration_stores_metadata():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)

    entry = integration.remember(
        context,
        "Tool execution succeeded.",
        metadata={
            "source": "runtime",
            "step_order": 1,
        },
    )

    assert entry.metadata == {
        "source": "runtime",
        "step_order": 1,
    }


def test_memory_integration_stores_explicit_source():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)

    entry = integration.remember(
        context,
        "The user supplied this information.",
        source="user",
    )

    assert entry.metadata == {
        "source": "user",
    }


def test_memory_integration_store_for_task_stores_explicit_source():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)

    entry = integration.store_for_task(
        context.task.id,
        "Manually recorded observation.",
        source="manual",
    )

    assert entry.metadata == {
        "source": "manual",
    }


def test_memory_integration_explicit_metadata_source_takes_precedence():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)

    entry = integration.remember(
        context,
        "Source metadata was explicitly provided.",
        source="user",
        metadata={
            "source": "observation",
            "step_order": 2,
        },
    )

    assert entry.metadata == {
        "source": "observation",
        "step_order": 2,
    }


def test_memory_integration_source_preserves_additional_metadata():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)

    entry = integration.remember(
        context,
        "Memory with explicit source and metadata.",
        source="inference",
        metadata={
            "importance": 10,
            "step_order": 3,
        },
    )

    assert entry.metadata == {
        "importance": 10,
        "step_order": 3,
        "source": "inference",
    }


def test_memory_integration_runtime_output_defaults_to_runtime_source():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)

    entry = integration.store_execution_output(
        context,
        "Execution completed successfully.",
    )

    assert entry.metadata == {
        "source": "runtime",
    }


def test_memory_integration_runtime_output_preserves_explicit_source():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)

    entry = integration.store_execution_output(
        context,
        "Observation was produced during execution.",
        metadata={
            "source": "observation",
            "step_order": 1,
        },
    )

    assert entry.metadata == {
        "source": "observation",
        "step_order": 1,
    }


def test_memory_integration_stores_technocore_observation():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)
    observation = make_observation()

    entry = integration.store_observation(
        context,
        observation,
    )

    assert entry.task_id == context.task.id
    assert entry.agent_id is None
    assert entry.content == observation.to_untrusted_text()
    assert entry.metadata == {
        "source": "observation",
        "room": "lobby",
        "since": 100,
        "message_count": 2,
        "first_sequence": 101,
        "last_sequence": 102,
        "observed_at": "2026-09-03T10:02:00+00:00",
    }


def test_memory_integration_stores_empty_technocore_observation():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)
    observation = make_observation(
        messages=(),
    )

    entry = integration.store_observation(
        context,
        observation,
    )

    assert entry.content == observation.to_untrusted_text()
    assert entry.metadata == {
        "source": "observation",
        "room": "lobby",
        "since": 100,
        "message_count": 0,
        "first_sequence": None,
        "last_sequence": None,
        "observed_at": "2026-09-03T10:02:00+00:00",
    }


def test_memory_integration_observation_preserves_additional_metadata():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)
    observation = make_observation()

    entry = integration.store_observation(
        context,
        observation,
        metadata={
            "importance": 10,
            "source": "custom-observation",
        },
    )

    assert entry.metadata == {
        "source": "custom-observation",
        "room": "lobby",
        "since": 100,
        "message_count": 2,
        "first_sequence": 101,
        "last_sequence": 102,
        "observed_at": "2026-09-03T10:02:00+00:00",
        "importance": 10,
    }


def test_memory_integration_rejects_invalid_observation():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)

    with pytest.raises(
        TypeError,
        match="observation must be a TechnocoreObservation",
    ):
        integration.store_observation(
            context,
            "not an observation",
        )


def test_memory_integration_observation_does_not_mutate_context():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)
    observation = make_observation()

    integration.store_observation(
        context,
        observation,
    )

    assert context.memories == ()


def test_memory_integration_rejects_non_string_source():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)

    with pytest.raises(
        TypeError,
        match="source must be a string or None",
    ):
        integration.remember(
            context,
            "Invalid source type.",
            source=123,
        )


def test_memory_integration_rejects_empty_source():
    context = make_context()
    store = InMemoryMemoryStore()

    integration = MemoryIntegration(store)

    with pytest.raises(
        ValueError,
        match="source must not be empty",
    ):
        integration.remember(
            context,
            "Invalid empty source.",
            source="   ",
        )


def test_memory_integration_filters_by_agent():
    context = make_context(agent_id="agent-1")
    store = InMemoryMemoryStore()

    matching = MemoryEntry(
        content="Matching agent memory",
        task_id=context.task.id,
        agent_id="agent-1",
    )

    other = MemoryEntry(
        content="Other agent memory",
        task_id=context.task.id,
        agent_id="agent-2",
    )

    store.store(matching)
    store.store(other)

    integration = MemoryIntegration(store)

    memories = integration.retrieve(context)

    assert memories == (matching,)


def test_memory_integration_does_not_mutate_context():
    context = make_context()
    store = InMemoryMemoryStore()

    entry = MemoryEntry(
        content="Existing memory",
        task_id=context.task.id,
    )

    store.store(entry)

    integration = MemoryIntegration(store)

    enriched = integration.enrich_context(context)

    assert context.memories == ()
    assert enriched.memories == (entry,)


def test_memory_integration_retrieves_successful_recovery_for_future_task():
    store = InMemoryMemoryStore()
    integration = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    recovery_context = make_context(
        agent_id="test-agent",
        description="Recover from repeated execution failure",
    )

    recovery_output = (
        "Recovered execution after repeated failure "
        "by using a memory-informed replanning strategy."
    )

    entry = integration.store_execution_output(
        recovery_context,
        recovery_output,
    )

    future_task = Task(
        description=(
            "Recover from repeated execution failure "
            "using a memory-informed replanning strategy"
        ),
    )

    relevant = integration.retrieve_relevant(
        future_task,
        agent_id="test-agent",
    )

    assert relevant == (entry,)
    assert relevant[0].content == recovery_output
    assert relevant[0].task_id == recovery_context.task.id
    assert relevant[0].agent_id == "test-agent"
    assert relevant[0].metadata == {
        "source": "runtime",
    }


def test_memory_integration_recovery_memory_is_excluded_from_own_historical_search():
    store = InMemoryMemoryStore()
    integration = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    context = make_context(
        agent_id="test-agent",
        description="Repeated failure recovery",
    )

    entry = integration.store_execution_output(
        context,
        "Repeated failure recovery succeeded.",
    )

    relevant = integration.retrieve_relevant(
        context.task,
        agent_id="test-agent",
    )

    assert entry not in relevant
    assert relevant == ()


def test_memory_integration_enriches_future_context_with_recovery_memory():
    store = InMemoryMemoryStore()
    integration = MemoryIntegration(
        store,
        agent_id="test-agent",
    )

    recovery_context = make_context(
        agent_id="test-agent",
        description="Recover execution after repeated failure",
    )

    entry = integration.store_execution_output(
        recovery_context,
        "Successful recovery using memory-informed replanning.",
    )

    future_context = make_context(
        agent_id="test-agent",
        description=(
            "Recover execution after repeated failure "
            "using memory-informed replanning"
        ),
    )

    enriched = integration.enrich_context(
        future_context,
    )

    assert enriched is not future_context
    assert enriched.task.id == future_context.task.id
    assert enriched.agent_id == "test-agent"
    assert enriched.memories == (entry,)
