from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.agent.loop import AgentLoop, AgentLoopResult
from src.agent.qualification import (
    QualificationCapability,
    QualificationDecision,
    QualificationResult,
    TaskQualifier,
)
from src.agent.task import Task
from src.agent.task_source import ObservedTask


class TaskSource(Protocol):
    def poll(self) -> tuple[ObservedTask, ...]:
        ...


class ProcessableTaskSource(TaskSource, Protocol):
    def mark_processed(self, message_id: int) -> None:
        ...


class AgentExecutor(Protocol):
    def run(self, task: Task) -> AgentLoopResult:
        ...


class ResultPublisher(Protocol):
    def publish(
        self,
        observed: ObservedTask,
        result: AgentLoopResult,
    ) -> object:
        ...


@dataclass(frozen=True, slots=True)
class AutonomousRun:
    discovered: tuple[ObservedTask, ...]
    results: tuple[AgentLoopResult, ...]
    qualifications: tuple[QualificationResult, ...] = ()

    @property
    def accepted_count(self) -> int:
        """Return the number of accepted tasks."""

        return sum(
            qualification.accepted
            for qualification in self.qualifications
        )

    @property
    def rejected_count(self) -> int:
        """Return the number of explicitly rejected tasks."""

        return sum(
            qualification.rejected
            for qualification in self.qualifications
        )

    @property
    def ignored_count(self) -> int:
        """Return the number of ignored tasks."""

        return sum(
            qualification.ignored
            for qualification in self.qualifications
        )


class AutonomousAgent:
    """Autonomous agent with a qualification boundary before execution."""

    def __init__(
        self,
        *,
        task_source: TaskSource,
        runtime: AgentExecutor | None = None,
        publisher: ResultPublisher | None = None,
        qualifier: TaskQualifier | None = None,
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

        if publisher is not None and not hasattr(
            publisher,
            "publish",
        ):
            raise TypeError(
                "publisher must provide a publish() method"
            )

        if qualifier is None:
            qualifier = TaskQualifier()

        if not hasattr(qualifier, "qualify"):
            raise TypeError(
                "qualifier must provide a qualify() method"
            )

        self.task_source = task_source
        self.runtime = runtime
        self.publisher = publisher
        self.qualifier = qualifier

    def run_once(self) -> AutonomousRun:
        """
        Poll, qualify, and execute externally sourced tasks.

        No task reaches the AgentLoop unless the qualification gate
        explicitly accepts it.
        """

        discovered = self.task_source.poll()

        if not isinstance(discovered, tuple):
            discovered = tuple(discovered)

        qualifications = tuple(
            self.qualifier.qualify(observed)
            for observed in discovered
        )

        results: list[AgentLoopResult] = []

        for observed, qualification in zip(
            discovered,
            qualifications,
        ):
            if qualification.decision != (
                QualificationDecision.ACCEPT
            ):
                self._acknowledge(observed.message_id)
                continue

            if qualification.task is None:
                raise RuntimeError(
                    "accepted qualification did not contain a task"
                )

            result = self.runtime.run(
                qualification.task,
                allowed_capabilities=self._capabilities_for(
                    qualification.capability
                ),
            )
            results.append(result)

            if not result.result.succeeded:
                continue

            if self.publisher is not None:
                self.publisher.publish(
                    observed,
                    result,
                )

            self._acknowledge(
                observed.message_id
            )

        return AutonomousRun(
            discovered=discovered,
            results=tuple(results),
            qualifications=qualifications,
        )

    def run_task(self, task: Task) -> AgentLoopResult:
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

        return self.runtime.run(task)

    @staticmethod
    def _capabilities_for(
        capability: QualificationCapability | None,
    ) -> frozenset[str]:
        if capability == QualificationCapability.CALCULATOR:
            return frozenset({"calculator"})

        raise PermissionError(
            "accepted task has no supported execution capability"
        )

    def _acknowledge(
        self,
        message_id: int,
    ) -> None:
        mark_processed = getattr(
            self.task_source,
            "mark_processed",
            None,
        )

        if mark_processed is None:
            return

        mark_processed(message_id)