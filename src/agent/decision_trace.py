from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4

from src.agent.decision import AutonomyAction


@dataclass(frozen=True, slots=True)
class AutonomyDecisionEvent:
    """Immutable record of one autonomy decision made by the agent."""

    sequence: int
    action: AutonomyAction
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    step_id: UUID | None = None
    trigger: str = "policy"
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, int):
            raise TypeError("sequence must be an integer")

        if self.sequence < 0:
            raise ValueError(
                "sequence must not be negative"
            )

        if not isinstance(self.action, AutonomyAction):
            raise TypeError(
                "action must be an AutonomyAction"
            )

        if not isinstance(self.reason, str):
            raise TypeError(
                "reason must be a string"
            )

        if not self.reason.strip():
            raise ValueError(
                "reason must not be empty"
            )

        if not isinstance(self.evidence, Mapping):
            raise TypeError(
                "evidence must be a mapping"
            )

        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(self.evidence)),
        )

        if self.step_id is not None:
            if not isinstance(self.step_id, UUID):
                raise TypeError(
                    "step_id must be a UUID or None"
                )

        if not isinstance(self.trigger, str):
            raise TypeError(
                "trigger must be a string"
            )

        if not self.trigger.strip():
            raise ValueError(
                "trigger must not be empty"
            )

        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                "timestamp must be a datetime"
            )

        if not isinstance(self.id, UUID):
            raise TypeError(
                "id must be a UUID"
            )