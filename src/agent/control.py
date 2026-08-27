from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ControlDecision(str, Enum):
    """Decision returned by the execution controller."""

    CONTINUE = "continue"
    STOP = "stop"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """Structured outcome produced after executing an execution step."""

    success: bool
    output: Any = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        """Return True when the step execution failed."""

        return not self.success


class ExecutionController:
    """Determines how the agent should proceed after a step."""

    def decide(self, outcome: StepOutcome) -> ControlDecision:
        """Return a control decision based on the step outcome."""

        if not isinstance(outcome, StepOutcome):
            raise TypeError("outcome must be a StepOutcome")

        if outcome.failed:
            return ControlDecision.FAIL

        return ControlDecision.CONTINUE