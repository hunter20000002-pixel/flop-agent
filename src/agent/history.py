from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from src.agent.control import ControlDecision
from src.agent.plan import ExecutionStep


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Record of a single execution step."""

    step_id: UUID
    description: str
    success: bool
    output: Any = None
    error: str | None = None
    decision: ControlDecision = ControlDecision.CONTINUE
    id: UUID = field(default_factory=uuid4)

    @property
    def failed(self) -> bool:
        """Return True when the step failed."""

        return not self.success


@dataclass(frozen=True, slots=True)
class ExecutionHistory:
    """Immutable ordered history of execution records."""

    task_id: UUID
    records: tuple[ExecutionRecord, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Return True when no execution records exist."""

        return not self.records

    @property
    def record_count(self) -> int:
        """Return the number of execution records."""

        return len(self.records)

    @property
    def last(self) -> ExecutionRecord | None:
        """Return the most recent execution record."""

        if not self.records:
            return None

        return self.records[-1]

    def add(self, record: ExecutionRecord) -> ExecutionHistory:
        """Return a new history with the supplied record appended."""

        if not isinstance(record, ExecutionRecord):
            raise TypeError("record must be an ExecutionRecord")

        return ExecutionHistory(
            task_id=self.task_id,
            records=self.records + (record,),
        )

    def record(
        self,
        step: ExecutionStep,
        *,
        success: bool,
        output: Any = None,
        error: str | None = None,
        decision: ControlDecision = ControlDecision.CONTINUE,
    ) -> ExecutionHistory:
        """Return a new history containing a record for the supplied step."""

        if not isinstance(step, ExecutionStep):
            raise TypeError("step must be an ExecutionStep")

        execution_record = ExecutionRecord(
            step_id=step.id,
            description=step.description,
            success=success,
            output=output,
            error=error,
            decision=decision,
        )

        return self.add(execution_record)

    @classmethod
    def from_step(
        cls,
        *,
        task_id: UUID,
        step: ExecutionStep,
        success: bool,
        output: Any = None,
        error: str | None = None,
        decision: ControlDecision = ControlDecision.CONTINUE,
    ) -> ExecutionHistory:
        """Create a history containing one record for a step."""
        
        if not isinstance(step, ExecutionStep):
            raise TypeError("step must be an ExecutionStep")
        return cls(

            task_id=task_id,
            records=(
                ExecutionRecord(
                    step_id=step.id,
                    description=step.description,
                    success=success,
                    output=output,
                    error=error,
                    decision=decision,
                ),
            ),
        )