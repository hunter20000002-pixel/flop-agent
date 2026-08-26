from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from src.agent.task import TaskStatus


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Structured result produced by the agent runtime."""

    task_id: UUID
    status: TaskStatus
    executed_steps: int
    output: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return True when execution completed successfully."""

        return self.status == TaskStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """Return True when execution failed."""

        return self.status == TaskStatus.FAILED