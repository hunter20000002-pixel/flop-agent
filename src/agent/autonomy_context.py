from __future__ import annotations

from dataclasses import dataclass

from src.agent.history import ExecutionHistory
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.result import ExecutionResult
from src.agent.task import Task

@dataclass(frozen=True, slots=True)
class AutonomyDecisionContext:
    """
    Immutable evidence available to the autonomy decision layer.

    AgentContext represents the agent's current execution state.

    AutonomyDecisionContext represents the evidence the autonomy layer
    can use when deciding what should happen next.
    """

    task: Task
    current_plan: ExecutionPlan | None
    current_step: ExecutionStep | None
    execution_history: ExecutionHistory
    last_result: ExecutionResult | None
    failure_count: int = 0
    retry_count: int = 0
    replan_count: int = 0
    allowed_capabilities: frozenset[str] | None = None
    remaining_step_budget: int | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the decision context."""

        if not isinstance(self.task, Task):
            raise TypeError("task must be a Task")

        if self.current_plan is not None and not isinstance(
            self.current_plan,
            ExecutionPlan,
        ):
            raise TypeError(
                "current_plan must be an ExecutionPlan or None"
            )

        if self.current_step is not None and not isinstance(
            self.current_step,
            ExecutionStep,
        ):
            raise TypeError(
                "current_step must be an ExecutionStep or None"
            )

        if not isinstance(
            self.execution_history,
            ExecutionHistory,
        ):
            raise TypeError(
                "execution_history must be an ExecutionHistory"
            )

        if self.last_result is not None and not isinstance(
            self.last_result,
            ExecutionResult,
        ):
            raise TypeError(
                "last_result must be an ExecutionResult or None"
            )

        self._validate_counter(
            self.failure_count,
            "failure_count",
        )
        self._validate_counter(
            self.retry_count,
            "retry_count",
        )
        self._validate_counter(
            self.replan_count,
            "replan_count",
        )

        if self.remaining_step_budget is not None:
            self._validate_counter(
                self.remaining_step_budget,
                "remaining_step_budget",
            )

        if self.allowed_capabilities is not None:
            normalized = frozenset(self.allowed_capabilities)

            for capability in normalized:
                if not isinstance(capability, str):
                    raise TypeError(
                        "allowed_capabilities must contain strings"
                    )

                if not capability.strip():
                    raise ValueError(
                        "allowed_capabilities must not contain "
                        "empty strings"
                    )

            object.__setattr__(
                self,
                "allowed_capabilities",
                normalized,
            )

    @staticmethod
    def _validate_counter(
        value: int,
        name: str,
    ) -> None:
        """Validate a non-negative integer counter."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be a non-negative integer"
            )

        if value < 0:
            raise ValueError(
                f"{name} must be a non-negative integer"
            )

    @property
    def task_id(self) -> str:
        """Return the identifier of the associated task."""

        return self.task.id

    @property
    def plan_steps(self) -> tuple[ExecutionStep, ...]:
        """Return the current plan's steps, or an empty tuple."""

        if self.current_plan is None:
            return ()

        return self.current_plan.steps

    @property
    def capabilities(self) -> frozenset[str] | None:
        """Return the authorized capabilities."""

        return self.allowed_capabilities

    @property
    def has_plan(self) -> bool:
        """Return True when a current execution plan exists."""

        return self.current_plan is not None

    @property
    def has_current_step(self) -> bool:
        """Return True when a current execution step exists."""

        return self.current_step is not None

    @property
    def has_last_result(self) -> bool:
        """Return True when a previous execution result exists."""

        return self.last_result is not None

    @property
    def has_failures(self) -> bool:
        """Return True when at least one failure has been observed."""

        return self.failure_count > 0

    @property
    def has_retries(self) -> bool:
        """Return True when at least one retry has occurred."""

        return self.retry_count > 0

    @property
    def has_replans(self) -> bool:
        """Return True when at least one replan has occurred."""

        return self.replan_count > 0

    @property
    def budget_exhausted(self) -> bool:
        """
        Return True when a remaining step budget exists and is zero.
        """

        return self.remaining_step_budget == 0

    def with_counters(
        self,
        *,
        failure_count: int | None = None,
        retry_count: int | None = None,
        replan_count: int | None = None,
    ) -> AutonomyDecisionContext:
        """
        Return a new context with updated decision counters.
        """

        return AutonomyDecisionContext(
            task=self.task,
            current_plan=self.current_plan,
            current_step=self.current_step,
            execution_history=self.execution_history,
            last_result=self.last_result,
            failure_count=(
                self.failure_count
                if failure_count is None
                else failure_count
            ),
            retry_count=(
                self.retry_count
                if retry_count is None
                else retry_count
            ),
            replan_count=(
                self.replan_count
                if replan_count is None
                else replan_count
            ),
            allowed_capabilities=self.allowed_capabilities,
            remaining_step_budget=self.remaining_step_budget,
        )

    def with_plan(
        self,
        plan: ExecutionPlan | None,
    ) -> AutonomyDecisionContext:
        """Return a new context with a different current plan."""

        return AutonomyDecisionContext(
            task=self.task,
            current_plan=plan,
            current_step=self.current_step,
            execution_history=self.execution_history,
            last_result=self.last_result,
            failure_count=self.failure_count,
            retry_count=self.retry_count,
            replan_count=self.replan_count,
            allowed_capabilities=self.allowed_capabilities,
            remaining_step_budget=self.remaining_step_budget,
        )

    def with_step(
        self,
        step: ExecutionStep | None,
    ) -> AutonomyDecisionContext:
        """Return a new context with a different current step."""

        return AutonomyDecisionContext(
            task=self.task,
            current_plan=self.current_plan,
            current_step=step,
            execution_history=self.execution_history,
            last_result=self.last_result,
            failure_count=self.failure_count,
            retry_count=self.retry_count,
            replan_count=self.replan_count,
            allowed_capabilities=self.allowed_capabilities,
            remaining_step_budget=self.remaining_step_budget,
        )

    def with_result(
        self,
        result: ExecutionResult | None,
    ) -> AutonomyDecisionContext:
        """Return a new context with a different last result."""

        return AutonomyDecisionContext(
            task=self.task,
            current_plan=self.current_plan,
            current_step=self.current_step,
            execution_history=self.execution_history,
            last_result=result,
            failure_count=self.failure_count,
            retry_count=self.retry_count,
            replan_count=self.replan_count,
            allowed_capabilities=self.allowed_capabilities,
            remaining_step_budget=self.remaining_step_budget,
        )

    def with_remaining_step_budget(
        self,
        remaining_step_budget: int | None,
    ) -> AutonomyDecisionContext:
        """
        Return a new context with an updated remaining step budget.
        """

        return AutonomyDecisionContext(
            task=self.task,
            current_plan=self.current_plan,
            current_step=self.current_step,
            execution_history=self.execution_history,
            last_result=self.last_result,
            failure_count=self.failure_count,
            retry_count=self.retry_count,
            replan_count=self.replan_count,
            allowed_capabilities=self.allowed_capabilities,
            remaining_step_budget=remaining_step_budget,
        )