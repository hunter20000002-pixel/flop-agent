
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4

import pytest

from src.agent.memory import (
    InMemoryStore,
    MemoryEntry,
    MemoryStore,
)


def test_memory_entry_stores_content():
    entry = MemoryEntry(
        content="The agent learned that 2 + 2 = 4.",
    )

    assert entry.content == "The agent learned that 2 + 2 = 4."
    assert isinstance(entry.id, UUID)


def test_memory_entry_supports_task_and_agent_association():
    task_id = uuid4()

    entry = MemoryEntry(
        content="Task-specific memory",
        task_id=task_id,
        agent_id="did:key:test-agent",
    )

    assert entry.task_id == task_id
    assert entry.agent_id == "did:key:test-agent"


def test_memory_entry_defaults_to_current_utc_time():
    before = datetime.now(timezone.utc)

    entry = MemoryEntry(
        content="Timestamped memory",
    )

    after = datetime.now(timezone.utc)

    assert before <= entry.created_at <= after
    assert entry.created_at.tzinfo == timezone.utc


def test_memory_entry_preserves_metadata():
    entry = MemoryEntry(
        content="Memory with metadata",
        metadata={
            "source": "test",
            "confidence": 0.95,
            "attempt": 1,
        },
    )

    assert entry.metadata["source"] == "test"
    assert entry.metadata["confidence"] == 0.95
    assert entry.metadata["attempt"] == 1


def test_memory_entry_metadata_is_copied():
    metadata = {
        "source": "test",
        "attempt": 1,
    }

    entry = MemoryEntry(
        content="Immutable metadata",
        metadata=metadata,
    )

    metadata["attempt"] = 999

    assert entry.metadata["attempt"] == 1


def test_memory_entry_is_immutable():
    entry = MemoryEntry(
        content="Immutable memory",
    )

    with pytest.raises(AttributeError):
        entry.content = "Changed"


def test_memory_entry_rejects_empty_content():
    with pytest.raises(
        ValueError,
        match="content must not be empty",
    ):
        MemoryEntry(content="   ")


def test_memory_entry_rejects_invalid_content_type():
    with pytest.raises(
        TypeError,
        match="content must be a string",
    ):
        MemoryEntry(content=123)


def test_memory_entry_rejects_invalid_task_id():
    with pytest.raises(
        TypeError,
        match="task_id must be a UUID or None",
    ):
        MemoryEntry(
            content="Invalid task",
            task_id="not-a-uuid",
        )


def test_memory_entry_rejects_invalid_agent_id():
    with pytest.raises(
        TypeError,
        match="agent_id must be a string or None",
    ):
        MemoryEntry(
            content="Invalid agent",
            agent_id=123,
        )


def test_memory_entry_rejects_empty_agent_id():
    with pytest.raises(
        ValueError,
        match="agent_id must not be empty",
    ):
        MemoryEntry(
            content="Invalid agent",
            agent_id="   ",
        )


def test_memory_entry_rejects_invalid_metadata():
    with pytest.raises(
        TypeError,
        match="metadata must be a mapping",
    ):
        MemoryEntry(
            content="Invalid metadata",
            metadata=["not", "a", "mapping"],
        )


def test_memory_entry_rejects_naive_datetime():
    with pytest.raises(
        ValueError,
        match="created_at must be timezone-aware",
    ):
        MemoryEntry(
            content="Naive timestamp",
            created_at=datetime(2026, 8, 28, 12, 0, 0),
        )


def test_memory_entry_matches_no_filters():
    entry = MemoryEntry(
        content="General memory",
    )

    assert entry.matches()


def test_memory_entry_matches_task_filter():
    task_id = uuid4()

    entry = MemoryEntry(
        content="Task memory",
        task_id=task_id,
    )

    assert entry.matches(task_id=task_id)
    assert not entry.matches(task_id=uuid4())


def test_memory_entry_matches_agent_filter():
    entry = MemoryEntry(
        content="Agent memory",
        agent_id="agent-1",
    )

    assert entry.matches(agent_id="agent-1")
    assert not entry.matches(agent_id="agent-2")


def test_memory_entry_matches_combined_filters():
    task_id = uuid4()

    entry = MemoryEntry(
        content="Specific memory",
        task_id=task_id,
        agent_id="agent-1",
    )

    assert entry.matches(
        task_id=task_id,
        agent_id="agent-1",
    )

    assert not entry.matches(
        task_id=task_id,
        agent_id="agent-2",
    )


def test_in_memory_store_implements_memory_store():
    store = InMemoryStore()

    assert isinstance(store, MemoryStore)


def test_empty_store():
    store = InMemoryStore()

    assert store.count == 0
    assert store.query() == ()


def test_store_and_retrieve_memory():
    store = InMemoryStore()

    entry = MemoryEntry(
        content="Stored memory",
    )

    returned = store.store(entry)

    assert returned is entry
    assert store.count == 1
    assert store.retrieve(entry.id) == entry


def test_store_preserves_insertion_order():
    store = InMemoryStore()

    first = MemoryEntry(content="First")
    second = MemoryEntry(content="Second")
    third = MemoryEntry(content="Third")

    store.store(first)
    store.store(second)
    store.store(third)

    assert store.query() == (
        first,
        second,
        third,
    )


def test_store_rejects_invalid_entry():
    store = InMemoryStore()

    with pytest.raises(
        TypeError,
        match="entry must be a MemoryEntry",
    ):
        store.store("not a memory")


def test_store_rejects_duplicate_entry_id():
    store = InMemoryStore()
    entry_id = uuid4()

    first = MemoryEntry(
        content="First",
        id=entry_id,
    )

    second = MemoryEntry(
        content="Second",
        id=entry_id,
    )

    store.store(first)

    with pytest.raises(
        ValueError,
        match="memory entry already exists",
    ):
        store.store(second)


def test_retrieve_missing_memory_returns_none():
    store = InMemoryStore()

    assert store.retrieve(uuid4()) is None


def test_retrieve_rejects_invalid_id():
    store = InMemoryStore()

    with pytest.raises(
        TypeError,
        match="entry_id must be a UUID",
    ):
        store.retrieve("not a uuid")


def test_delete_existing_memory():
    store = InMemoryStore()

    entry = MemoryEntry(
        content="Delete me",
    )

    store.store(entry)

    assert store.delete(entry.id) is True
    assert store.retrieve(entry.id) is None
    assert store.count == 0


def test_delete_missing_memory_returns_false():
    store = InMemoryStore()

    assert store.delete(uuid4()) is False


def test_delete_rejects_invalid_id():
    store = InMemoryStore()

    with pytest.raises(
        TypeError,
        match="entry_id must be a UUID",
    ):
        store.delete("not a uuid")


def test_query_by_task_id():
    store = InMemoryStore()

    task_a = uuid4()
    task_b = uuid4()

    first = MemoryEntry(
        content="Task A memory 1",
        task_id=task_a,
    )

    second = MemoryEntry(
        content="Task B memory",
        task_id=task_b,
    )

    third = MemoryEntry(
        content="Task A memory 2",
        task_id=task_a,
    )

    store.store(first)
    store.store(second)
    store.store(third)

    assert store.query(task_id=task_a) == (
        first,
        third,
    )


def test_query_by_agent_id():
    store = InMemoryStore()

    first = MemoryEntry(
        content="Agent 1 memory",
        agent_id="agent-1",
    )

    second = MemoryEntry(
        content="Agent 2 memory",
        agent_id="agent-2",
    )

    third = MemoryEntry(
        content="Another Agent 1 memory",
        agent_id="agent-1",
    )

    store.store(first)
    store.store(second)
    store.store(third)

    assert store.query(agent_id="agent-1") == (
        first,
        third,
    )


def test_query_by_task_and_agent():
    store = InMemoryStore()

    task_id = uuid4()

    first = MemoryEntry(
        content="Matching memory",
        task_id=task_id,
        agent_id="agent-1",
    )

    second = MemoryEntry(
        content="Wrong agent",
        task_id=task_id,
        agent_id="agent-2",
    )

    third = MemoryEntry(
        content="Wrong task",
        task_id=uuid4(),
        agent_id="agent-1",
    )

    store.store(first)
    store.store(second)
    store.store(third)

    assert store.query(
        task_id=task_id,
        agent_id="agent-1",
    ) == (first,)


def test_query_rejects_invalid_task_id():
    store = InMemoryStore()

    with pytest.raises(
        TypeError,
        match="task_id must be a UUID or None",
    ):
        store.query(task_id="not a uuid")


def test_query_rejects_invalid_agent_id_type():
    store = InMemoryStore()

    with pytest.raises(
        TypeError,
        match="agent_id must be a string or None",
    ):
        store.query(agent_id=123)


def test_query_rejects_empty_agent_id():
    store = InMemoryStore()

    with pytest.raises(
        ValueError,
        match="agent_id must not be empty",
    ):
        store.query(agent_id="   ")


def test_clear_removes_all_memories():
    store = InMemoryStore()

    store.store(MemoryEntry(content="First"))
    store.store(MemoryEntry(content="Second"))

    assert store.count == 2

    store.clear()

    assert store.count == 0
    assert store.query() == ()


def test_memory_entries_can_have_explicit_timestamps():
    timestamp = datetime(
        2026,
        8,
        28,
        10,
        0,
        0,
        tzinfo=timezone.utc,
    )

    entry = MemoryEntry(
        content="Historical memory",
        created_at=timestamp,
    )

    assert entry.created_at == timestamp


def test_memory_entries_preserve_subsecond_timestamps():
    timestamp = datetime(
        2026,
        8,
        28,
        10,
        0,
        0,
        500000,
        tzinfo=timezone.utc,
    )

    entry = MemoryEntry(
        content="Precise timestamp",
        created_at=timestamp,
    )

    assert entry.created_at == timestamp


def test_query_returns_immutable_tuple():
    store = InMemoryStore()

    entry = MemoryEntry(
        content="Tuple result",
    )

    store.store(entry)

    result = store.query()

    assert isinstance(result, tuple)

    with pytest.raises(AttributeError):
        result.append(entry)


def test_store_does_not_change_memory_entry():
    store = InMemoryStore()

    entry = MemoryEntry(
        content="Stable memory",
        metadata={"source": "test"},
    )

    original_id = entry.id
    original_content = entry.content

    returned = store.store(entry)

    assert returned.id == original_id
    assert returned.content == original_content
    assert returned is entry


def test_delete_does_not_affect_other_memories():
    store = InMemoryStore()

    first = MemoryEntry(content="First")
    second = MemoryEntry(content="Second")

    store.store(first)
    store.store(second)

    store.delete(first.id)

    assert store.retrieve(first.id) is None
    assert store.retrieve(second.id) == second
    assert store.count == 1
