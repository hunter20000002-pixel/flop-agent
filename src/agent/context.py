from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from src.agent.history import ExecutionHistory, ExecutionRecord
from src.agent.memory import MemoryEntry
from src.agent.plan import ExecutionPlan
from src.agent.task import Task


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Immutable execution context available to an agent."""

    task: Task
    plan: ExecutionPlan | None = None
    history: ExecutionHistory | None = None
    memories: tuple[MemoryEntry, ...] = ()
    agent_id: str | None = None
    state: str = "idle"
    allowed_capabilities: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, Task):
            raise TypeError("task must be a Task")

        if self.plan is not None and not isinstance(
            self.plan,
            ExecutionPlan,
        ):
            raise TypeError("plan must be an ExecutionPlan or None")

        if self.history is not None and not isinstance(
            self.history,
            ExecutionHistory,
        ):
            raise TypeError(
                "history must be an ExecutionHistory or None"
            )

        if not isinstance(self.memories, tuple):
            raise TypeError("memories must be a tuple")

        for memory in self.memories:
            if not isinstance(memory, MemoryEntry):
                raise TypeError(
                    "memories must contain only MemoryEntry objects"
                )

        if self.agent_id is not None:
            if not isinstance(self.agent_id, str):
                raise TypeError(
                    "agent_id must be a string or None"
                )

            if not self.agent_id.strip():
                raise ValueError("agent_id must not be empty")

        if not isinstance(self.state, str):
            raise TypeError("state must be a string")

        if not self.state.strip():
            raise ValueError("state must not be empty")

        if self.allowed_capabilities is not None:
            if not isinstance(
                self.allowed_capabilities,
                frozenset,
            ):
                raise TypeError(
                    "allowed_capabilities must be a frozenset or None"
                )

            for capability in self.allowed_capabilities:
                if not isinstance(capability, str):
                    raise TypeError(
                        "allowed_capabilities must contain only strings"
                    )

                if not capability.strip():
                    raise ValueError(
                        "allowed_capabilities cannot contain empty strings"
                    )

    @property
    def task_id(self) -> UUID:
        """Return the ID of the current task."""

        return self.task.id

    @property
    def plan_steps(self) -> tuple:
        """Return the planned execution steps."""

        if self.plan is None:
            return ()

        return self.plan.steps

    @property
    def history_records(self) -> tuple[ExecutionRecord, ...]:
        """Return execution history records."""

        if self.history is None:
            return ()

        return self.history.records

    @property
    def last_execution(self) -> ExecutionRecord | None:
        """Return the most recent execution record."""

        if self.history is None:
            return None

        return self.history.last

    @property
    def successful_executions(
        self,
    ) -> tuple[ExecutionRecord, ...]:
        """Return all successful execution records."""

        if self.history is None:
            return ()

        return self.history.successful_records

    @property
    def failed_executions(
        self,
    ) -> tuple[ExecutionRecord, ...]:
        """Return all failed execution records."""

        if self.history is None:
            return ()

        return self.history.failed_records

    @property
    def has_execution_failures(self) -> bool:
        """Return whether the execution history contains failures."""

        if self.history is None:
            return False

        return self.history.has_failures

    @property
    def memory_count(self) -> int:
        """Return the number of available memories."""

        return len(self.memories)

    @property
    def has_memories(self) -> bool:
        """Return whether any memories are available."""

        return bool(self.memories)

    def memory_for_id(
        self,
        memory_id: UUID,
    ) -> MemoryEntry | None:
        """Return a memory by ID, if present."""

        if not isinstance(memory_id, UUID):
            raise TypeError("memory_id must be a UUID")

        for memory in self.memories:
            if memory.id == memory_id:
                return memory

        return None

    def with_plan(
        self,
        plan: ExecutionPlan | None,
    ) -> AgentContext:
        """Return a new context with an updated plan."""

        return AgentContext(
            task=self.task,
            plan=plan,
            history=self.history,
            memories=self.memories,
            agent_id=self.agent_id,
            state=self.state,
            allowed_capabilities=self.allowed_capabilities,
        )

    def with_history(
        self,
        history: ExecutionHistory | None,
    ) -> AgentContext:
        """Return a new context with updated execution history."""

        return AgentContext(
            task=self.task,
            plan=self.plan,
            history=history,
            memories=self.memories,
            agent_id=self.agent_id,
            state=self.state,
            allowed_capabilities=self.allowed_capabilities,
        )

    def with_memories(
        self,
        memories: tuple[MemoryEntry, ...],
    ) -> AgentContext:
        """Return a new context with updated memories."""

        return AgentContext(
            task=self.task,
            plan=self.plan,
            history=self.history,
            memories=memories,
            agent_id=self.agent_id,
            state=self.state,
            allowed_capabilities=self.allowed_capabilities,
        )

    def with_state(
        self,
        state: str,
    ) -> AgentContext:
        """Return a new context with updated agent state."""

        return AgentContext(
            task=self.task,
            plan=self.plan,
            history=self.history,
            memories=self.memories,
            agent_id=self.agent_id,
            state=state,
            allowed_capabilities=self.allowed_capabilities,
        )

    def with_allowed_capabilities(
        self,
        capabilities: Iterable[str] | None,
    ) -> AgentContext:
        """Return a new context with updated capability authorization."""

        normalized = (
            None
            if capabilities is None
            else frozenset(capabilities)
        )

        return AgentContext(
            task=self.task,
            plan=self.plan,
            history=self.history,
            memories=self.memories,
            agent_id=self.agent_id,
            state=self.state,
            allowed_capabilities=normalized,
        )