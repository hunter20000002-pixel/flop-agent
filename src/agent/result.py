from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from src.agent.history import ExecutionHistory
from src.agent.task import TaskStatus

if TYPE_CHECKING:
    from src.agent.goal import GoalVerificationResult


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    task_id: UUID
    status: TaskStatus
    executed_steps: int = 0
    output: Any = None
    error: str | None = None
    history: ExecutionHistory | None = None
    progress_made: bool | None = None
    goal_verification: GoalVerificationResult | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    @property
    def failed(self) -> bool:
        return self.status == TaskStatus.FAILED