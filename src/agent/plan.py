from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class ExecutionStep:
    """A single step in an agent execution plan."""

    description: str
    order: int
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("step description cannot be empty")

        if self.order < 1:
            raise ValueError("step order must be greater than zero")


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """An ordered set of steps associated with a task."""

    task_id: UUID
    steps: tuple[ExecutionStep, ...] = ()

    def __post_init__(self) -> None:
        orders = [step.order for step in self.steps]

        if len(orders) != len(set(orders)):
            raise ValueError("step orders must be unique")

        if orders != sorted(orders):
            raise ValueError("steps must be ordered by step order")

    @property
    def is_empty(self) -> bool:
        """Return True when the plan contains no execution steps."""

        return not self.steps

    @property
    def step_count(self) -> int:
        """Return the number of execution steps."""

        return len(self.steps)