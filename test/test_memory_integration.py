from src.agent.context import AgentContext
from src.agent.memory import InMemoryMemoryStore, MemoryEntry
from src.agent.memory_integration import MemoryIntegration
from src.agent.task import Task


def make_context(
    *,
    agent_id: str | None = None,
) -> AgentContext:
    task = Task(description="Memory integration test")

    return AgentContext(
        task=task,
        agent_id=agent_id,
        state="running",
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