from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from src.agent.autonomy_context import AutonomyDecisionContext
from src.agent.context import AgentContext
from src.agent.control import ControlDecision


class AutonomyAction(str, Enum):
    """Actions the autonomy layer can request."""

    EXECUTE = "execute"
    RETRY = "retry"
    REPLAN = "replan"
    STOP = "stop"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class AutonomyDecision:
    """Structured decision produced by the autonomy policy."""

    action: AutonomyAction
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze decision evidence."""

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

    @property
    def should_execute(self) -> bool:
        """Return True when execution should proceed."""

        return self.action == AutonomyAction.EXECUTE

    @property
    def should_retry(self) -> bool:
        """Return True when the previous execution should be retried."""

        return self.action == AutonomyAction.RETRY

    @property
    def should_replan(self) -> bool:
        """Return True when a new plan should be created."""

        return self.action == AutonomyAction.REPLAN

    @property
    def should_stop(self) -> bool:
        """Return True when execution should stop."""

        return self.action == AutonomyAction.STOP

    @property
    def should_complete(self) -> bool:
        """Return True when the task should be considered complete."""

        return self.action == AutonomyAction.COMPLETE


class AutonomyPolicy:
    """Policy responsible for deciding the agent's next action."""

    def decide(
        self,
        context: AgentContext | AutonomyDecisionContext,
    ) -> AutonomyDecision:
        """
        Decide what the agent should do next.

        AgentContext remains supported for backward compatibility with
        existing callers.

        AutonomyDecisionContext is used by AgentRuntime when runtime-owned
        execution evidence such as failure, retry, replan, progress, and
        remaining step-budget counters is available.
        """

        if isinstance(context, AutonomyDecisionContext):
            return self._decide_from_autonomy_context(context)

        if isinstance(context, AgentContext):
            return self._decide_from_agent_context(context)

        raise TypeError(
            "context must be an AgentContext"
        )

    @staticmethod
    def _evidence_from_autonomy_context(
        context: AutonomyDecisionContext,
    ) -> dict[str, Any]:
        """Build structured evidence from runtime decision context."""

        return {
            "task_id": context.task.id,
            "progress_made": (
                context.last_result.progress_made
                if context.last_result is not None
                else None
            ),
            "failure_count": context.failure_count,
            "retry_count": context.retry_count,
            "replan_count": context.replan_count,
            "remaining_step_budget": context.remaining_step_budget,
            "current_step": (
                context.current_step.id
                if context.current_step is not None
                else None
            ),
        }

    @staticmethod
    def _evidence_from_agent_context(
        context: AgentContext,
    ) -> dict[str, Any]:
        """Build compatibility evidence from AgentContext."""

        return {
            "task_id": context.task.id,
            "progress_made": None,
            "failure_count": (
                1
                if (
                    context.last_execution is not None
                    and context.last_execution.failed
                )
                else 0
            ),
            "retry_count": 0,
            "replan_count": 0,
            "remaining_step_budget": None,
            "current_step": (
                context.plan.steps[0].id
                if context.plan is not None
                and context.plan.steps
                else None
            ),
        }

    def _decide_from_autonomy_context(
        self,
        context: AutonomyDecisionContext,
    ) -> AutonomyDecision:
        """Apply autonomy policy using runtime-owned decision evidence."""

        evidence = self._evidence_from_autonomy_context(context)

        if context.budget_exhausted:
            return AutonomyDecision(
                action=AutonomyAction.STOP,
                reason="execution step budget is exhausted",
                evidence=evidence,
            )

        if context.current_plan is None:
            return AutonomyDecision(
                action=AutonomyAction.REPLAN,
                reason="no execution plan is available",
                evidence=evidence,
            )

        if not context.plan_steps:
            return AutonomyDecision(
                action=AutonomyAction.COMPLETE,
                reason="execution plan contains no steps",
                evidence=evidence,
            )

        if context.last_result is not None:
            if context.last_result.failed:
                if context.failure_count >= 2:
                    return AutonomyDecision(
                        action=AutonomyAction.REPLAN,
                        reason=(
                            "repeated execution failures require "
                            "a new plan"
                        ),
                        evidence=evidence,
                    )

                return AutonomyDecision(
                    action=AutonomyAction.RETRY,
                    reason="most recent execution failed",
                    evidence=evidence,
                )

            if context.last_result.progress_made is False:
                return AutonomyDecision(
                    action=AutonomyAction.REPLAN,
                    reason="most recent execution made no progress",
                    evidence=evidence,
                )

        return AutonomyDecision(
            action=AutonomyAction.EXECUTE,
            reason="executable plan is available",
            evidence=evidence,
        )

    def _decide_from_agent_context(
        self,
        context: AgentContext,
    ) -> AutonomyDecision:
        """Apply the legacy AgentContext-based autonomy policy."""

        evidence = self._evidence_from_agent_context(context)
        state = context.state.strip().lower()

        if state == "completed":
            return AutonomyDecision(
                action=AutonomyAction.COMPLETE,
                reason="task is already completed",
                evidence=evidence,
            )

        if state in {"stopped", "cancelled"}:
            return AutonomyDecision(
                action=AutonomyAction.STOP,
                reason=f"task is already {state}",
                evidence=evidence,
            )

        if context.plan is None:
            return AutonomyDecision(
                action=AutonomyAction.REPLAN,
                reason="no execution plan is available",
                evidence=evidence,
            )

        if not context.plan_steps:
            return AutonomyDecision(
                action=AutonomyAction.COMPLETE,
                reason="execution plan contains no steps",
                evidence=evidence,
            )

        if context.last_execution is not None:
            if context.last_execution.decision == ControlDecision.STOP:
                return AutonomyDecision(
                    action=AutonomyAction.STOP,
                    reason="controller requested a stop",
                    evidence=evidence,
                )

            if context.last_execution.decision == ControlDecision.FAIL:
                return AutonomyDecision(
                    action=AutonomyAction.RETRY,
                    reason="most recent execution failed",
                    evidence=evidence,
                )

        return AutonomyDecision(
            action=AutonomyAction.EXECUTE,
            reason="executable plan is available",
            evidence=evidence,
        )