from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4


class TaskStatus(str, Enum):
    """Lifecycle states for an autonomous-agent task."""

    PENDING = "pending"
    PLANNING = "planning"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Task:
    """A unit of work submitted to the autonomous agent."""

    description: str
    id: UUID = field(default_factory=uuid4)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def set_status(self, status: TaskStatus) -> None:
        """Update the task lifecycle state."""

        if not isinstance(status, TaskStatus):
            raise TypeError("status must be a TaskStatus")

        self.status = status
        self.updated_at = datetime.now(timezone.utc)

    def mark_planning(self) -> None:
        """Move the task into the planning phase."""

        self.set_status(TaskStatus.PLANNING)

    def mark_ready(self) -> None:
        """Mark the task as ready for execution."""

        self.set_status(TaskStatus.READY)

    def mark_running(self) -> None:
        """Mark the task as currently executing."""

        self.set_status(TaskStatus.RUNNING)

    def mark_completed(self) -> None:
        """Mark the task as successfully completed."""

        self.set_status(TaskStatus.COMPLETED)

    def mark_failed(self) -> None:
        """Mark the task as failed."""

        self.set_status(TaskStatus.FAILED)

    def mark_cancelled(self) -> None:
        """Mark the task as cancelled."""

        self.set_status(TaskStatus.CANCELLED)