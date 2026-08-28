from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID

from src.agent.control import ControlDecision
from src.agent.history import ExecutionHistory, ExecutionRecord


class SQLiteExecutionHistoryStore:
    """Persistent SQLite-backed execution history store."""

    def __init__(self, path: str | Path) -> None:
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
        """Create the execution history schema."""

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                description TEXT NOT NULL,
                success INTEGER NOT NULL,
                output TEXT,
                error TEXT,
                decision TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )

        self._connection.commit()

    def append(
        self,
        record: ExecutionRecord,
    ) -> ExecutionRecord:
        """Persist one execution record."""

        if not isinstance(record, ExecutionRecord):
            raise TypeError(
                "record must be an ExecutionRecord"
            )

        metadata = self._serialize_metadata(
            record.metadata,
        )

        self._connection.execute(
            """
            INSERT INTO execution_history (
                task_id,
                step_id,
                description,
                success,
                output,
                error,
                decision,
                started_at,
                completed_at,
                metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.task_id),
                str(record.step_id),
                record.description,
                int(record.success),
                self._serialize_value(record.output),
                record.error,
                record.decision.value,
                record.started_at.isoformat(),
                record.completed_at.isoformat(),
                metadata,
            ),
        )

        self._connection.commit()

        return record

    def get_for_task(
        self,
        task_id: UUID,
    ) -> ExecutionHistory:
        """Return persisted history for a task."""

        if not isinstance(task_id, UUID):
            raise TypeError(
                "task_id must be a UUID"
            )

        rows = self._connection.execute(
            """
            SELECT
                step_id,
                description,
                success,
                output,
                error,
                decision,
                started_at,
                completed_at,
                metadata
            FROM execution_history
            WHERE task_id = ?
            ORDER BY id ASC
            """,
            (str(task_id),),
        ).fetchall()

        history = ExecutionHistory(
            task_id=task_id,
        )

        for row in rows:
            record = self._deserialize_record(
                task_id,
                row,
            )

            history = history.add(record)

        return history

    def count(
        self,
        task_id: UUID | None = None,
    ) -> int:
        """Return the number of stored execution records."""

        if task_id is not None and not isinstance(
            task_id,
            UUID,
        ):
            raise TypeError(
                "task_id must be a UUID or None"
            )

        if task_id is None:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM execution_history
                """
            ).fetchone()
        else:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM execution_history
                WHERE task_id = ?
                """,
                (str(task_id),),
            ).fetchone()

        return int(row["count"])

    def delete_for_task(
        self,
        task_id: UUID,
    ) -> bool:
        """Delete all execution history for a task."""

        if not isinstance(task_id, UUID):
            raise TypeError(
                "task_id must be a UUID"
            )

        cursor = self._connection.execute(
            """
            DELETE FROM execution_history
            WHERE task_id = ?
            """,
            (str(task_id),),
        )

        self._connection.commit()

        return cursor.rowcount > 0

    def clear(self) -> None:
        """Delete all persisted execution history."""

        self._connection.execute(
            """
            DELETE FROM execution_history
            """
        )

        self._connection.commit()

    def close(self) -> None:
        """Close the SQLite connection."""

        self._connection.close()

    def __enter__(
        self,
    ) -> SQLiteExecutionHistoryStore:
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    @staticmethod
    def _serialize_value(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return str(value)

    @staticmethod
    def _serialize_metadata(
        metadata: Any,
    ) -> str:
        if not metadata:
            return "{}"

        return repr(dict(metadata))

    @staticmethod
    def _deserialize_metadata(
        value: str,
    ) -> dict[str, Any]:
        if not value or value == "{}":
            return {}

        return {}

    @classmethod
    def _deserialize_record(
        cls,
        task_id: UUID,
        row: sqlite3.Row,
    ) -> ExecutionRecord:
        return ExecutionRecord(
            task_id=task_id,
            step_id=UUID(row["step_id"]),
            description=row["description"],
            success=bool(row["success"]),
            output=row["output"],
            error=row["error"],
            decision=ControlDecision(
                row["decision"],
            ),
            started_at=cls._parse_datetime(
                row["started_at"],
            ),
            completed_at=cls._parse_datetime(
                row["completed_at"],
            ),
            metadata=cls._deserialize_metadata(
                row["metadata"],
            ),
        )

    @staticmethod
    def _parse_datetime(
        value: str,
    ):
        from datetime import datetime

        return datetime.fromisoformat(value)