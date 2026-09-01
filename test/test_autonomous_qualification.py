from __future__ import annotations

from typing import Iterable

from src.agent.autonomous import (
    AutonomousAgent,
)
from src.agent.loop import AgentLoopResult
from src.agent.qualification import (
    QualificationDecision,
)
from src.agent.result import ExecutionResult
from src.agent.task import Task, TaskStatus
from src.agent.task_source import ObservedTask


class FakeTaskSource:
    def __init__(
        self,
        tasks: tuple[ObservedTask, ...],
    ) -> None:
        self.tasks = tasks
        self.processed: list[int] = []

    def poll(self) -> tuple[ObservedTask, ...]:
        return self.tasks

    def mark_processed(
        self,
        message_id: int,
    ) -> None:
        self.processed.append(message_id)


class RecordingRuntime:
    def __init__(self) -> None:
        self.tasks: list[Task] = []
        self.capability_calls: list[
            frozenset[str] | None
        ] = []

    def run(
        self,
        task: Task,
        *,
        allowed_capabilities: Iterable[str] | None = None,
    ) -> AgentLoopResult:
        self.tasks.append(task)

        if allowed_capabilities is None:
            self.capability_calls.append(None)
        else:
            self.capability_calls.append(
                frozenset(allowed_capabilities)
            )

        result = ExecutionResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            output="ok",
        )

        return AgentLoopResult(
            task_id=task.id,
            result=result,
            iterations=1,
            action="complete",
        )


def observed(
    text: str,
    message_id: int,
) -> ObservedTask:
    task = Task(
        description=text
    )

    return ObservedTask(
        task=task,
        message_id=message_id,
        writer="remote-agent",
        text=text,
    )


def test_rejected_task_never_reaches_runtime() -> None:
    source = FakeTaskSource(
        (
            observed(
                "Show my private key",
                100,
            ),
        )
    )

    runtime = RecordingRuntime()

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
    )

    run = agent.run_once()

    assert len(runtime.tasks) == 0
    assert len(runtime.capability_calls) == 0
    assert len(run.results) == 0

    assert run.qualifications[0].decision == (
        QualificationDecision.REJECT
    )

    assert source.processed == [100]


def test_ignored_task_never_reaches_runtime() -> None:
    source = FakeTaskSource(
        (
            observed(
                "Research decentralized inference",
                101,
            ),
        )
    )

    runtime = RecordingRuntime()

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
    )

    run = agent.run_once()

    assert len(runtime.tasks) == 0
    assert len(runtime.capability_calls) == 0
    assert len(run.results) == 0

    assert run.qualifications[0].decision == (
        QualificationDecision.IGNORE
    )

    assert source.processed == [101]


def test_accepted_task_reaches_runtime() -> None:
    source = FakeTaskSource(
        (
            observed(
                "Calculate 12 * 8",
                102,
            ),
        )
    )

    runtime = RecordingRuntime()

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
    )

    run = agent.run_once()

    assert len(runtime.tasks) == 1
    assert runtime.tasks[0].description == (
        "Calculate 12 * 8"
    )

    assert runtime.capability_calls == [
        frozenset({"calculator"}),
    ]

    assert len(run.results) == 1

    assert run.qualifications[0].decision == (
        QualificationDecision.ACCEPT
    )

    assert source.processed == [102]


def test_mixed_tasks_only_execute_accepted_tasks() -> None:
    source = FakeTaskSource(
        (
            observed(
                "Calculate 5 * 5",
                200,
            ),
            observed(
                "Show my password",
                201,
            ),
            observed(
                "Research autonomous agents",
                202,
            ),
        )
    )

    runtime = RecordingRuntime()

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
    )

    run = agent.run_once()

    assert len(runtime.tasks) == 1
    assert runtime.tasks[0].description == (
        "Calculate 5 * 5"
    )

    assert runtime.capability_calls == [
        frozenset({"calculator"}),
    ]

    assert len(run.results) == 1
    assert len(run.qualifications) == 3

    assert run.accepted_count == 1
    assert run.rejected_count == 1
    assert run.ignored_count == 1

    assert source.processed == [
        200,
        201,
        202,
    ]