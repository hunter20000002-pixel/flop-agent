from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """Immutable piece of information stored in agent memory."""

    content: str
    task_id: UUID | None = None
    agent_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")

        if not self.content.strip():
            raise ValueError("content must not be empty")

        if self.task_id is not None and not isinstance(
            self.task_id,
            UUID,
        ):
            raise TypeError("task_id must be a UUID or None")

        if self.agent_id is not None and not isinstance(
            self.agent_id,
            str,
        ):
            raise TypeError("agent_id must be a string or None")

        if self.agent_id is not None and not self.agent_id.strip():
            raise ValueError("agent_id must not be empty")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime")

        if self.created_at.tzinfo is None:
            raise ValueError(
                "created_at must be timezone-aware"
            )

        if not isinstance(self.id, UUID):
            raise TypeError("id must be a UUID")

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    def matches(
        self,
        *,
        task_id: UUID | None = None,
        agent_id: str | None = None,
    ) -> bool:
        """Return True when this entry matches the supplied filters."""

        if task_id is not None and self.task_id != task_id:
            return False

        if agent_id is not None and self.agent_id != agent_id:
            return False

        return True


class MemoryStore(ABC):
    """Abstract interface for agent memory storage."""

    @abstractmethod
    def store(self, entry: MemoryEntry) -> MemoryEntry:
        """Store a memory entry and return it."""

        raise NotImplementedError

    @abstractmethod
    def retrieve(self, entry_id: UUID) -> MemoryEntry | None:
        """Retrieve an entry by ID."""

        raise NotImplementedError

    @abstractmethod
    def delete(self, entry_id: UUID) -> bool:
        """Delete an entry by ID."""

        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        *,
        task_id: UUID | None = None,
        agent_id: str | None = None,
    ) -> tuple[MemoryEntry, ...]:
        """Return entries matching the supplied filters."""

        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored memories."""

        raise NotImplementedError


class InMemoryStore(MemoryStore):
    """Deterministic in-memory implementation of MemoryStore."""

    def __init__(self) -> None:
        self._entries: dict[UUID, MemoryEntry] = {}

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        """Store an immutable memory entry."""

        if not isinstance(entry, MemoryEntry):
            raise TypeError("entry must be a MemoryEntry")

        if entry.id in self._entries:
            raise ValueError(
                f"memory entry already exists: {entry.id}"
            )

        self._entries[entry.id] = entry

        return entry

    def retrieve(self, entry_id: UUID) -> MemoryEntry | None:
        """Retrieve an entry by its UUID."""

        if not isinstance(entry_id, UUID):
            raise TypeError("entry_id must be a UUID")

        return self._entries.get(entry_id)

    def delete(self, entry_id: UUID) -> bool:
        """Delete an entry and return whether it existed."""

        if not isinstance(entry_id, UUID):
            raise TypeError("entry_id must be a UUID")

        if entry_id not in self._entries:
            return False

        del self._entries[entry_id]

        return True

    def query(
        self,
        *,
        task_id: UUID | None = None,
        agent_id: str | None = None,
    ) -> tuple[MemoryEntry, ...]:
        """Return matching entries in insertion order."""

        if task_id is not None and not isinstance(task_id, UUID):
            raise TypeError("task_id must be a UUID or None")

        if agent_id is not None and not isinstance(agent_id, str):
            raise TypeError("agent_id must be a string or None")

        if agent_id is not None and not agent_id.strip():
            raise ValueError("agent_id must not be empty")

        return tuple(
            entry
            for entry in self._entries.values()
            if entry.matches(
                task_id=task_id,
                agent_id=agent_id,
            )
        )

    def clear(self) -> None:
        """Remove all stored entries."""

        self._entries.clear()

    @property
    def count(self) -> int:
        """Return the number of stored entries."""

        return len(self._entries)


class InMemoryMemoryStore(InMemoryStore):
    """Backward-compatible name for the in-memory memory store."""

    pass


class SQLiteMemoryStore(MemoryStore):
    """Persistent SQLite implementation of MemoryStore."""

    def __init__(self, database_path: str) -> None:
        if not isinstance(database_path, str):
            raise TypeError("database_path must be a string")

        if not database_path.strip():
            raise ValueError("database_path must not be empty")

        self.database_path = database_path
        self._connection = sqlite3.connect(database_path)

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                task_id TEXT,
                agent_id TEXT,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        self._connection.commit()

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        """Persist a memory entry."""

        if not isinstance(entry, MemoryEntry):
            raise TypeError("entry must be a MemoryEntry")

        try:
            self._connection.execute(
                """
                INSERT INTO memories (
                    id,
                    content,
                    task_id,
                    agent_id,
                    metadata,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(entry.id),
                    entry.content,
                    str(entry.task_id)
                    if entry.task_id is not None
                    else None,
                    entry.agent_id,
                    json.dumps(dict(entry.metadata)),
                    entry.created_at.isoformat(),
                ),
            )

            self._connection.commit()

        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"memory entry already exists: {entry.id}"
            ) from exc

        return entry

    def retrieve(self, entry_id: UUID) -> MemoryEntry | None:
        """Retrieve a memory entry by UUID."""

        if not isinstance(entry_id, UUID):
            raise TypeError("entry_id must be a UUID")

        row = self._connection.execute(
            """
            SELECT
                id,
                content,
                task_id,
                agent_id,
                metadata,
                created_at
            FROM memories
            WHERE id = ?
            """,
            (str(entry_id),),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_entry(row)

    def delete(self, entry_id: UUID) -> bool:
        """Delete a memory entry."""

        if not isinstance(entry_id, UUID):
            raise TypeError("entry_id must be a UUID")

        cursor = self._connection.execute(
            """
            DELETE FROM memories
            WHERE id = ?
            """,
            (str(entry_id),),
        )

        self._connection.commit()

        return cursor.rowcount > 0

    def query(
        self,
        *,
        task_id: UUID | None = None,
        agent_id: str | None = None,
    ) -> tuple[MemoryEntry, ...]:
        """Return matching memories in insertion order."""

        if task_id is not None and not isinstance(task_id, UUID):
            raise TypeError("task_id must be a UUID or None")

        if agent_id is not None and not isinstance(agent_id, str):
            raise TypeError("agent_id must be a string or None")

        if agent_id is not None and not agent_id.strip():
            raise ValueError("agent_id must not be empty")

        rows = self._connection.execute(
            """
            SELECT
                id,
                content,
                task_id,
                agent_id,
                metadata,
                created_at
            FROM memories
            ORDER BY rowid ASC
            """
        ).fetchall()

        entries = tuple(
            self._row_to_entry(row)
            for row in rows
        )

        return tuple(
            entry
            for entry in entries
            if entry.matches(
                task_id=task_id,
                agent_id=agent_id,
            )
        )

    def clear(self) -> None:
        """Remove all persisted memories."""

        self._connection.execute(
            "DELETE FROM memories"
        )

        self._connection.commit()

    @property
    def count(self) -> int:
        """Return the number of persisted memories."""

        row = self._connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()

        return int(row[0])

    def close(self) -> None:
        """Close the SQLite connection."""

        self._connection.close()

    def __enter__(self) -> SQLiteMemoryStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()

    @staticmethod
    def _row_to_entry(
        row: tuple[Any, ...],
    ) -> MemoryEntry:
        """Convert a SQLite row into a MemoryEntry."""

        (
            entry_id,
            content,
            task_id,
            agent_id,
            metadata,
            created_at,
        ) = row

        return MemoryEntry(
            id=UUID(entry_id),
            content=content,
            task_id=UUID(task_id) if task_id else None,
            agent_id=agent_id,
            metadata=json.loads(metadata),
            created_at=datetime.fromisoformat(created_at),
        )