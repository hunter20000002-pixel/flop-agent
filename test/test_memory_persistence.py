from __future__ import annotations

from uuid import uuid4

from src.agent.memory import (
    MemoryEntry,
    SQLiteMemoryStore,
)


def test_memory_persists_across_store_reopen(tmp_path):
    database = tmp_path / "memory.db"

    entry = MemoryEntry(
        content="persistent memory",
        task_id=uuid4(),
        agent_id="agent-1",
        metadata={
            "source": "test",
            "importance": 10,
        },
    )

    first_store = SQLiteMemoryStore(str(database))

    first_store.store(entry)
    first_store.close()

    second_store = SQLiteMemoryStore(str(database))

    restored = second_store.retrieve(entry.id)

    assert restored == entry
    assert restored is not None
    assert restored.content == "persistent memory"
    assert restored.task_id == entry.task_id
    assert restored.agent_id == "agent-1"
    assert dict(restored.metadata) == {
        "source": "test",
        "importance": 10,
    }

    second_store.close()


def test_memory_query_persists_across_reopen(tmp_path):
    database = tmp_path / "memory.db"

    task_id = uuid4()

    first_store = SQLiteMemoryStore(str(database))

    first = MemoryEntry(
        content="first memory",
        task_id=task_id,
        agent_id="agent-1",
    )

    second = MemoryEntry(
        content="second memory",
        task_id=uuid4(),
        agent_id="agent-2",
    )

    first_store.store(first)
    first_store.store(second)
    first_store.close()

    second_store = SQLiteMemoryStore(str(database))

    results = second_store.query(task_id=task_id)

    assert results == (first,)

    second_store.close()


def test_memory_delete_persists_across_reopen(tmp_path):
    database = tmp_path / "memory.db"

    entry = MemoryEntry(
        content="temporary persistent memory",
    )

    first_store = SQLiteMemoryStore(str(database))

    first_store.store(entry)

    assert first_store.delete(entry.id)

    first_store.close()

    second_store = SQLiteMemoryStore(str(database))

    assert second_store.retrieve(entry.id) is None
    assert second_store.count == 0

    second_store.close()


def test_memory_clear_persists_across_reopen(tmp_path):
    database = tmp_path / "memory.db"

    first_store = SQLiteMemoryStore(str(database))

    first_store.store(
        MemoryEntry(content="memory one")
    )

    first_store.store(
        MemoryEntry(content="memory two")
    )

    assert first_store.count == 2

    first_store.clear()
    first_store.close()

    second_store = SQLiteMemoryStore(str(database))

    assert second_store.count == 0
    assert second_store.query() == ()

    second_store.close()


def test_memory_store_context_manager(tmp_path):
    database = tmp_path / "memory.db"

    entry = MemoryEntry(
        content="context manager memory"
    )

    with SQLiteMemoryStore(str(database)) as store:
        store.store(entry)

    with SQLiteMemoryStore(str(database)) as store:
        restored = store.retrieve(entry.id)

    assert restored == entry


def test_memory_filters_by_agent_after_reopen(tmp_path):
    database = tmp_path / "memory.db"

    first_store = SQLiteMemoryStore(str(database))

    agent_one = MemoryEntry(
        content="agent one memory",
        agent_id="agent-1",
    )

    agent_two = MemoryEntry(
        content="agent two memory",
        agent_id="agent-2",
    )

    first_store.store(agent_one)
    first_store.store(agent_two)
    first_store.close()

    second_store = SQLiteMemoryStore(str(database))

    results = second_store.query(
        agent_id="agent-1"
    )

    assert results == (agent_one,)

    second_store.close()


def test_memory_count_persists_across_reopen(tmp_path):
    database = tmp_path / "memory.db"

    first_store = SQLiteMemoryStore(str(database))

    first_store.store(
        MemoryEntry(content="one")
    )

    first_store.store(
        MemoryEntry(content="two")
    )

    assert first_store.count == 2

    first_store.close()

    second_store = SQLiteMemoryStore(str(database))

    assert second_store.count == 2

    second_store.close()