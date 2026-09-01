from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.agent.loop import AgentLoop, AgentLoopResult
from src.agent.task import Task
from src.agent.task_source import ObservedTask


class TaskSource(Protocol):
    """Interface for external sources capable of discovering tasks."""

    def poll(self) -> tuple[ObservedTask, ...]:
        """Discover currently available tasks."""
        ...


class ProcessableTaskSource(TaskSource, Protocol):
    """Task source capable of acknowledging successful processing."""

    def mark_processed(
        self,
        message_id: int,
    ) -> None:
        """Acknowledge successful processing of a discovered task."""
        ...


class AgentExecutor(Protocol):
    """Interface for components capable of executing agent tasks."""

    def run(
        self,
        task: Task,
    ) -> AgentLoopResult:
        """Execute a task and return its autonomous loop result."""
        ...


@dataclass(frozen=True, slots=True)
class AutonomousRun:
    """Result of one autonomous observation/execution cycle."""

    discovered: tuple[ObservedTask, ...]
    results: tuple[AgentLoopResult, ...]


class AutonomousAgent:
    """Connects external task discovery to autonomous task execution."""

    def __init__(
        self,
        *,
        task_source: TaskSource,
        runtime: AgentExecutor | None = None,
    ) -> None:
        if not hasattr(task_source, "poll"):
            raise TypeError(
                "task_source must provide a poll() method"
            )

        if runtime is None:
            runtime = AgentLoop()

        if not hasattr(runtime, "run"):
            raise TypeError(
                "runtime must provide a run() method"
            )

        self.task_source = task_source
        self.runtime = runtime

    def run_once(self) -> AutonomousRun:
        """
        Perform one complete autonomous cycle.

        The cycle is:

        observe
        → discover tasks
        → execute tasks
        → acknowledge successful tasks
        """

        discovered = self.task_source.poll()

        results: list[AgentLoopResult] = []

        for observed in discovered:
            result = self.runtime.run(
                observed.task
            )

            results.append(result)

            if result.result.succeeded:
                self._acknowledge(
                    observed.message_id
                )

        return AutonomousRun(
            discovered=discovered,
            results=tuple(results),
        )

    def run_task(
        self,
        task: Task,
    ) -> AgentLoopResult:
        """Execute a task directly through the configured runtime."""

        if not isinstance(task, Task):
            raise TypeError(
                "task must be a Task"
            )

        return self.runtime.run(task)

    def _acknowledge(
        self,
        message_id: int,
    ) -> None:
        """Acknowledge a task when the source supports persistence."""

        mark_processed = getattr(
            self.task_source,
            "mark_processed",
            None,
        )

        if mark_processed is None:
            return

        mark_processed(message_id)