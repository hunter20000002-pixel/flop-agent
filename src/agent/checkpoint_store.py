
from __future__ import annotations

import sqlite3
from pathlib import Path


class SQLiteTaskCheckpointStore:
    """Persistent SQLite-backed store for Technocore task checkpoints."""

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._connection = sqlite3.connect(
            self.path,
        )

        self._connection.row_factory = sqlite3.Row

        self._create_schema()

    def _create_schema(self) -> None:
        """Create the checkpoint schema."""

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS task_checkpoints (
                source TEXT PRIMARY KEY,
                since INTEGER NOT NULL
            )
            """
        )

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_tasks (
                source TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                PRIMARY KEY (source, message_id)
            )
            """
        )

        self._connection.commit()

    def get_since(
        self,
        source: str,
        default: int = 0,
    ) -> int:
        """Return the persisted message cursor for a source."""

        if not isinstance(source, str):
            raise TypeError(
                "source must be a string"
            )

        if not source.strip():
            raise ValueError(
                "source cannot be empty"
            )

        if not isinstance(default, int):
            raise TypeError(
                "default must be an integer"
            )

        if default < 0:
            raise ValueError(
                "default cannot be negative"
            )

        row = self._connection.execute(
            """
            SELECT since
            FROM task_checkpoints
            WHERE source = ?
            """,
            (source,),
        ).fetchone()

        if row is None:
            return default

        return int(row["since"])

    def set_since(
        self,
        source: str,
        since: int,
    ) -> None:
        """Persist the latest observed message cursor."""

        if not isinstance(source, str):
            raise TypeError(
                "source must be a string"
            )

        if not source.strip():
            raise ValueError(
                "source cannot be empty"
            )

        if not isinstance(since, int):
            raise TypeError(
                "since must be an integer"
            )

        if since < 0:
            raise ValueError(
                "since cannot be negative"
            )

        self._connection.execute(
            """
            INSERT INTO task_checkpoints (
                source,
                since
            )
            VALUES (?, ?)
            ON CONFLICT(source)
            DO UPDATE SET since = excluded.since
            """,
            (
                source,
                since,
            ),
        )

        self._connection.commit()

    def mark_processed(
        self,
        source: str,
        message_id: int,
    ) -> bool:
        """Mark a message as processed.

        Return True when this is the first time the message was marked,
        and False when the message was already processed.
        """

        if not isinstance(source, str):
            raise TypeError(
                "source must be a string"
            )

        if not source.strip():
            raise ValueError(
                "source cannot be empty"
            )

        if not isinstance(message_id, int):
            raise TypeError(
                "message_id must be an integer"
            )

        if message_id < 0:
            raise ValueError(
                "message_id cannot be negative"
            )

        cursor = self._connection.execute(
            """
            INSERT OR IGNORE INTO processed_tasks (
                source,
                message_id
            )
            VALUES (?, ?)
            """,
            (
                source,
                message_id,
            ),
        )

        self._connection.commit()

        return cursor.rowcount == 1

    def is_processed(
        self,
        source: str,
        message_id: int,
    ) -> bool:
        """Return True when a message has already been processed."""

        if not isinstance(source, str):
            raise TypeError(
                "source must be a string"
            )

        if not source.strip():
            raise ValueError(
                "source cannot be empty"
            )

        if not isinstance(message_id, int):
            raise TypeError(
                "message_id must be an integer"
            )

        if message_id < 0:
            raise ValueError(
                "message_id cannot be negative"
            )

        row = self._connection.execute(
            """
            SELECT 1
            FROM processed_tasks
            WHERE source = ?
              AND message_id = ?
            """,
            (
                source,
                message_id,
            ),
        ).fetchone()

        return row is not None

    def close(self) -> None:
        """Close the SQLite connection."""

        self._connection.close()

    def __enter__(
        self,
    ) -> SQLiteTaskCheckpointStore:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
