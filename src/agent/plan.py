from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class ExecutionStep:
    """A single step in an agent execution plan."""

    description: str
    order: int
    id: UUID = field(default_factory=uuid4)
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("step description cannot be empty")

        if self.order < 1:
            raise ValueError("step order must be greater than zero")

        if self.tool_name is not None and not self.tool_name.strip():
            raise ValueError("tool name cannot be empty")

        object.__setattr__(self, "tool_args", dict(self.tool_args))

    @property
    def uses_tool(self) -> bool:
        """Return True when this step explicitly requests a tool."""

        return self.tool_name is not None


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