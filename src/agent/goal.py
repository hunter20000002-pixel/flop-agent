from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.result import ExecutionResult
from src.agent.task import Task


@dataclass(frozen=True, slots=True)
class GoalVerificationResult:
    """Result of verifying whether a task's goal was satisfied."""

    satisfied: bool
    reason: str
    evidence: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.satisfied, bool):
            raise TypeError("satisfied must be a boolean")

        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")

        if not self.reason.strip():
            raise ValueError("reason must not be empty")


class GoalVerifier:
    """Interface for verifying whether task execution satisfied its goal."""

    def verify(
        self,
        task: Task,
        result: ExecutionResult,
    ) -> GoalVerificationResult:
        """Verify whether an execution result satisfied the task goal."""

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

        if not isinstance(result, ExecutionResult):
            raise TypeError(
                "result must be an ExecutionResult"
            )

        if result.task_id != task.id:
            raise ValueError(
                "result.task_id must match task.id"
            )

        return self._verify(task, result)

    def _verify(
        self,
        task: Task,
        result: ExecutionResult,
    ) -> GoalVerificationResult:
        """Perform goal verification.

        Subclasses should override this method with task-specific
        verification logic.
        """

        return GoalVerificationResult(
            satisfied=result.succeeded,
            reason=(
                "execution result indicates success"
                if result.succeeded
                else "execution result indicates failure"
            ),
            evidence={
                "task_id": str(task.id),
                "status": result.status.value,
                "executed_steps": result.executed_steps,
            },
        )