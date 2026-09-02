from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from src.agent.history import ExecutionHistory
from src.agent.task import TaskStatus


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Structured result produced by an agent execution."""

    task_id: UUID
    status: TaskStatus
    executed_steps: int = 0
    output: Any = None
    error: str | None = None
    history: ExecutionHistory | None = None
    progress_made: bool | None = None

    @property
    def succeeded(self) -> bool:
        """Return True when execution completed successfully."""

        return self.status == TaskStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """Return True when execution failed."""

        return self.status == TaskStatus.FAILED