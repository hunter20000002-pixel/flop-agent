from __future__ import annotations

from dataclasses import dataclass

from src.agent.history import ExecutionHistory
from src.agent.memory import MemoryEntry
from src.agent.observation import TechnocoreObservation
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.result import ExecutionResult
from src.agent.task import Task


@dataclass(frozen=True, slots=True)
class AutonomyDecisionContext:
    """
    Immutable evidence available to the autonomy decision layer.

    AgentContext represents the agent's current execution state.

    AutonomyDecisionContext represents the richer runtime evidence
    available to the autonomy layer when deciding what should happen
    next.

    Compatibility properties expose the commonly used AgentContext
    attributes so existing autonomy policies can continue to operate
    against the new decision context.
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
    memories: tuple[MemoryEntry, ...] = ()

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

        if not isinstance(self.memories, tuple):
            raise TypeError(
                "memories must be a tuple"
            )

        for memory in self.memories:
            if not isinstance(memory, MemoryEntry):
                raise TypeError(
                    "memories must contain only MemoryEntry objects"
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
        """Return the steps in the current execution plan."""

        if self.current_plan is None:
            return ()

        return self.current_plan.steps

    @property
    def capabilities(self) -> frozenset[str] | None:
        """Return the capabilities authorized for this execution."""

        return self.allowed_capabilities

    @property
    def memory_count(self) -> int:
        """Return the number of memories available to autonomy."""

        return len(self.memories)

    @property
    def has_memories(self) -> bool:
        """Return True when autonomy has memory evidence available."""

        return bool(self.memories)

    @property
    def last_observation(self) -> TechnocoreObservation | None:
        """
        Return the most recent structured Technocore observation.

        Observation data is carried through ExecutionResult.data.
        Only a validated TechnocoreObservation is exposed here.
        """

        if self.last_result is None:
            return None

        if isinstance(
            self.last_result.data,
            TechnocoreObservation,
        ):
            return self.last_result.data

        return None

    @property
    def has_observation(self) -> bool:
        """Return True when the latest execution produced an observation."""

        return self.last_observation is not None

    @property
    def observation_message_count(self) -> int:
        """Return the number of messages in the latest observation."""

        observation = self.last_observation

        if observation is None:
            return 0

        return observation.message_count

    @property
    def observation_room(self) -> str | None:
        """Return the room associated with the latest observation."""

        observation = self.last_observation

        if observation is None:
            return None

        return observation.room

    @property
    def observation_first_sequence(self) -> int | None:
        """Return the first observed message sequence, when available."""

        observation = self.last_observation

        if observation is None:
            return None

        return observation.first_sequence

    @property
    def observation_last_sequence(self) -> int | None:
        """Return the last observed message sequence, when available."""

        observation = self.last_observation

        if observation is None:
            return None

        return observation.last_sequence

    @property
    def has_plan(self) -> bool:
        """Return True when an execution plan is available."""

        return self.current_plan is not None

    @property
    def has_current_step(self) -> bool:
        """Return True when a current execution step is available."""

        return self.current_step is not None

    @property
    def has_last_result(self) -> bool:
        """Return True when an execution result is available."""

        return self.last_result is not None

    @property
    def has_failures(self) -> bool:
        """Return True when at least one execution failure has occurred."""

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
    def goal_verification_failed(self) -> bool:
        """
        Return True when execution completed but semantic goal
        verification failed.
        """

        return (
            self.last_result is not None
            and self.last_result.goal_verification is not None
            and not self.last_result.goal_verification.satisfied
        )

    @property
    def goal_verification_succeeded(self) -> bool:
        """
        Return True when a configured goal verifier confirmed
        the task goal.
        """

        return (
            self.last_result is not None
            and self.last_result.goal_verification is not None
            and self.last_result.goal_verification.satisfied
        )

    @property
    def budget_exhausted(self) -> bool:
        """Return True when no execution attempts remain."""

        return self.remaining_step_budget == 0

    @property
    def plan(self) -> ExecutionPlan | None:
        """Compatibility alias for AgentContext.plan."""

        return self.current_plan

    @property
    def history(self) -> ExecutionHistory:
        """Compatibility alias for AgentContext.history."""

        return self.execution_history

    @property
    def last_execution(self):
        """
        Compatibility view of the most recent execution record.

        Older autonomy policies used AgentContext.last_execution.
        The runtime decision context now derives that value directly
        from the immutable execution history.
        """

        return self.execution_history.last

    @property
    def state(self) -> str:
        """
        Compatibility state representation.

        Runtime autonomy decisions operate primarily on explicit
        evidence fields rather than AgentContext state. Returning
        'running' preserves the behavior expected by legacy policies
        while a task is being evaluated by the runtime.
        """

        return "running"

    @property
    def agent_id(self):
        """Compatibility placeholder for AgentContext.agent_id."""

        return None

    def with_counters(
        self,
        *,
        failure_count: int | None = None,
        retry_count: int | None = None,
        replan_count: int | None = None,
    ) -> AutonomyDecisionContext:
        """Return a copy with updated autonomy counters."""

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
            memories=self.memories,
        )

    def with_plan(
        self,
        plan: ExecutionPlan | None,
    ) -> AutonomyDecisionContext:
        """Return a copy with a replacement execution plan."""

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
            memories=self.memories,
        )

    def with_step(
        self,
        step: ExecutionStep | None,
    ) -> AutonomyDecisionContext:
        """Return a copy with a replacement current step."""

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
            memories=self.memories,
        )

    def with_result(
        self,
        result: ExecutionResult | None,
    ) -> AutonomyDecisionContext:
        """Return a copy with a replacement execution result."""

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
            memories=self.memories,
        )

    def with_remaining_step_budget(
        self,
        remaining_step_budget: int | None,
    ) -> AutonomyDecisionContext:
        """Return a copy with an updated remaining step budget."""

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
            memories=self.memories,
        )

    def with_memories(
        self,
        memories: tuple[MemoryEntry, ...],
    ) -> AutonomyDecisionContext:
        """Return a copy with updated autonomy memories."""

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
            remaining_step_budget=self.remaining_step_budget,
            memories=memories,
        )