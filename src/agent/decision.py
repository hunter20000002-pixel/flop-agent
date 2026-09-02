from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
        execution evidence such as failure, retry, replan, and remaining
        step-budget counters is available.
        """

        if isinstance(context, AutonomyDecisionContext):
            return self._decide_from_autonomy_context(context)

        if isinstance(context, AgentContext):
            return self._decide_from_agent_context(context)

        raise TypeError(
            "context must be an AgentContext"
        )

    def _decide_from_autonomy_context(
        self,
        context: AutonomyDecisionContext,
    ) -> AutonomyDecision:
        """Apply autonomy policy using runtime-owned decision evidence."""

        if context.budget_exhausted:
            return AutonomyDecision(
                action=AutonomyAction.STOP,
                reason="execution step budget is exhausted",
            )

        if context.current_plan is None:
            return AutonomyDecision(
                action=AutonomyAction.REPLAN,
                reason="no execution plan is available",
            )

        if not context.plan_steps:
            return AutonomyDecision(
                action=AutonomyAction.COMPLETE,
                reason="execution plan contains no steps",
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
                    )

                return AutonomyDecision(
                    action=AutonomyAction.RETRY,
                    reason="most recent execution failed",
                )

        return AutonomyDecision(
            action=AutonomyAction.EXECUTE,
            reason="executable plan is available",
        )

    def _decide_from_agent_context(
        self,
        context: AgentContext,
    ) -> AutonomyDecision:
        """Apply the legacy AgentContext-based autonomy policy."""

        state = context.state.strip().lower()

        if state == "completed":
            return AutonomyDecision(
                action=AutonomyAction.COMPLETE,
                reason="task is already completed",
            )

        if state in {"stopped", "cancelled"}:
            return AutonomyDecision(
                action=AutonomyAction.STOP,
                reason=f"task is already {state}",
            )

        if context.plan is None:
            return AutonomyDecision(
                action=AutonomyAction.REPLAN,
                reason="no execution plan is available",
            )

        if not context.plan_steps:
            return AutonomyDecision(
                action=AutonomyAction.COMPLETE,
                reason="execution plan contains no steps",
            )

        if context.last_execution is not None:
            if context.last_execution.decision == ControlDecision.STOP:
                return AutonomyDecision(
                    action=AutonomyAction.STOP,
                    reason="controller requested a stop",
                )

            if context.last_execution.decision == ControlDecision.FAIL:
                return AutonomyDecision(
                    action=AutonomyAction.RETRY,
                    reason="most recent execution failed",
                )

        return AutonomyDecision(
            action=AutonomyAction.EXECUTE,
            reason="executable plan is available",
        )