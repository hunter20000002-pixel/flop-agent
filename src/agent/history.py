from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
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
    metadata: Mapping[str, Any] = field(default_factory=dict)
    capability: str | None = None
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

        if not isinstance(self.step_id, UUID):
            raise TypeError("step_id must be a UUID")

        if not isinstance(self.description, str):
            raise TypeError("description must be a string")

        if not self.description.strip():
            raise ValueError("description must not be empty")

        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean")

        if not isinstance(self.decision, ControlDecision):
            raise TypeError(
                "decision must be a ControlDecision"
            )

        if self.capability is not None:
            if not isinstance(self.capability, str):
                raise TypeError(
                    "capability must be a string or None"
                )

            if not self.capability.strip():
                raise ValueError(
                    "capability must not be empty"
                )

        if not isinstance(self.started_at, datetime):
            raise TypeError("started_at must be a datetime")

        if not isinstance(self.completed_at, datetime):
            raise TypeError("completed_at must be a datetime")

        if self.completed_at < self.started_at:
            raise ValueError(
                "completed_at cannot be earlier than started_at"
            )

        if self.success and self.error is not None:
            raise ValueError(
                "successful execution records cannot contain an error"
            )

        if not self.success and self.error is None:
            raise ValueError(
                "failed execution records must contain an error"
            )

    @property
    def failed(self) -> bool:
        """Return True when the step failed."""

        return not self.success

    @property
    def duration_seconds(self) -> float:
        """Return execution duration in seconds."""

        return (
            self.completed_at - self.started_at
        ).total_seconds()


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

    @property
    def successful_records(
        self,
    ) -> tuple[ExecutionRecord, ...]:
        """Return all successful execution records in execution order."""

        return tuple(
            record
            for record in self.records
            if record.success
        )

    @property
    def failed_records(
        self,
    ) -> tuple[ExecutionRecord, ...]:
        """Return all failed execution records in execution order."""

        return tuple(
            record
            for record in self.records
            if record.failed
        )

    @property
    def has_failures(self) -> bool:
        """Return True when at least one execution record failed."""

        return any(
            record.failed
            for record in self.records
        )

    @property
    def total_duration_seconds(self) -> float:
        """Return the total duration of all recorded executions."""

        return sum(
            record.duration_seconds
            for record in self.records
        )

    @property
    def capabilities_used(self) -> frozenset[str]:
        """
        Return the distinct capabilities used by this history.

        Records without a capability, such as ordinary executor or
        inference steps, are ignored.
        """

        return frozenset(
            record.capability
            for record in self.records
            if record.capability is not None
        )

    @property
    def tool_names_used(self) -> frozenset[str]:
        """
        Return the distinct tool names recorded in execution metadata.
        """

        return frozenset(
            tool_name
            for record in self.records
            if (
                tool_name := record.metadata.get("tool_name")
            ) is not None
            and isinstance(tool_name, str)
        )

    def records_for_step(
        self,
        step_id: UUID,
    ) -> tuple[ExecutionRecord, ...]:
        """Return all records associated with a step ID."""

        if not isinstance(step_id, UUID):
            raise TypeError("step_id must be a UUID")

        return tuple(
            record
            for record in self.records
            if record.step_id == step_id
        )

    def records_for_tool(
        self,
        tool_name: str,
    ) -> tuple[ExecutionRecord, ...]:
        """
        Return all records associated with a tool name.

        Tool identity is stored in execution metadata so that
        ExecutionRecord remains independent of the Tool abstraction.
        """

        if not isinstance(tool_name, str):
            raise TypeError("tool_name must be a string")

        if not tool_name.strip():
            raise ValueError("tool_name must not be empty")

        return tuple(
            record
            for record in self.records
            if record.metadata.get("tool_name") == tool_name
        )

    def records_for_capability(
        self,
        capability: str,
    ) -> tuple[ExecutionRecord, ...]:
        """Return all records associated with a capability."""

        if not isinstance(capability, str):
            raise TypeError("capability must be a string")

        if not capability.strip():
            raise ValueError(
                "capability must not be empty"
            )

        return tuple(
            record
            for record in self.records
            if record.capability == capability
        )

    def add(
        self,
        record: ExecutionRecord,
    ) -> ExecutionHistory:
        """Return a new history with the supplied record appended."""

        if not isinstance(record, ExecutionRecord):
            raise TypeError(
                "record must be an ExecutionRecord"
            )

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
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        capability: str | None = None,
    ) -> ExecutionHistory:
        """Return a new history containing a record for the supplied step."""

        if not isinstance(step, ExecutionStep):
            raise TypeError(
                "step must be an ExecutionStep"
            )

        execution_record = ExecutionRecord(
            step_id=step.id,
            description=step.description,
            success=success,
            output=output,
            error=error,
            decision=decision,
            metadata=metadata or {},
            capability=capability,
            started_at=(
                started_at
                if started_at is not None
                else datetime.now(timezone.utc)
            ),
            completed_at=(
                completed_at
                if completed_at is not None
                else datetime.now(timezone.utc)
            ),
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
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        capability: str | None = None,
    ) -> ExecutionHistory:
        """Create a history containing one record for a step."""

        if not isinstance(step, ExecutionStep):
            raise TypeError(
                "step must be an ExecutionStep"
            )

        execution_record = ExecutionRecord(
            step_id=step.id,
            description=step.description,
            success=success,
            output=output,
            error=error,
            decision=decision,
            metadata=metadata or {},
            capability=capability,
            started_at=(
                started_at
                if started_at is not None
                else datetime.now(timezone.utc)
            ),
            completed_at=(
                completed_at
                if completed_at is not None
                else datetime.now(timezone.utc)
            ),
        )

        return cls(
            task_id=task_id,
            records=(execution_record,),
        )