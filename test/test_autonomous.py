from __future__ import annotations

from typing import Iterable

from src.agent.autonomous import AutonomousAgent
from src.agent.decision import AutonomyAction
from src.agent.loop import AgentLoopResult
from src.agent.qualification import (
    QualificationCapability,
    QualificationDecision,
    QualificationResult,
)
from src.agent.result import ExecutionResult
from src.agent.task import Task, TaskStatus
from src.agent.task_source import ObservedTask


def make_observed_task(
    *,
    message_id: int = 100,
    text: str = "Calculate 10 + 5",
    writer: str = "agent-a",
) -> ObservedTask:
    """Create an externally observed task."""

    return ObservedTask(
        task=Task(
            description=text,
        ),
        message_id=message_id,
        writer=writer,
        text=text,
    )


def make_success_result(
    task: Task,
) -> AgentLoopResult:
    """Create a successful AgentLoop result."""

    return AgentLoopResult(
        task_id=task.id,
        result=ExecutionResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            executed_steps=1,
            output="15",
            error=None,
        ),
        iterations=1,
        action=AutonomyAction.COMPLETE,
    )


def make_failure_result(
    task: Task,
) -> AgentLoopResult:
    """Create a failed AgentLoop result."""

    return AgentLoopResult(
        task_id=task.id,
        result=ExecutionResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            executed_steps=1,
            output=None,
            error="execution failed",
        ),
        iterations=1,
        action=AutonomyAction.STOP,
    )


class FakeTaskSource:
    def __init__(
        self,
        tasks: tuple[ObservedTask, ...],
        *,
        events: list[str] | None = None,
    ) -> None:
        self.tasks = tasks
        self.poll_count = 0
        self.processed: list[int] = []
        self.events = events

    def poll(self) -> tuple[ObservedTask, ...]:
        self.poll_count += 1

        if self.events is not None:
            self.events.append("poll")

        return self.tasks

    def mark_processed(
        self,
        message_id: int,
    ) -> None:
        self.processed.append(message_id)

        if self.events is not None:
            self.events.append(
                f"ack:{message_id}"
            )


class FlakyTaskSource:
    """A task source that keeps returning an unacknowledged task."""

    def __init__(
        self,
        task: ObservedTask,
        *,
        events: list[str] | None = None,
    ) -> None:
        self.task = task
        self.poll_count = 0
        self.processed: list[int] = []
        self.events = events

    def poll(self) -> tuple[ObservedTask, ...]:
        self.poll_count += 1

        if self.events is not None:
            self.events.append("poll")

        if self.task.message_id in self.processed:
            return ()

        return (self.task,)

    def mark_processed(
        self,
        message_id: int,
    ) -> None:
        self.processed.append(message_id)

        if self.events is not None:
            self.events.append(
                f"ack:{message_id}"
            )


class FakeRuntime:
    def __init__(
        self,
        *,
        succeeded: bool = True,
        events: list[str] | None = None,
    ) -> None:
        self.succeeded = succeeded
        self.calls: list[Task] = []
        self.capability_calls: list[
            frozenset[str] | None
        ] = []
        self.events = events

    def run(
        self,
        task: Task,
        *,
        allowed_capabilities: Iterable[str] | None = None,
    ) -> AgentLoopResult:
        self.calls.append(task)

        if allowed_capabilities is None:
            self.capability_calls.append(None)
        else:
            self.capability_calls.append(
                frozenset(allowed_capabilities)
            )

        if self.events is not None:
            self.events.append("execute")

        if self.succeeded:
            return make_success_result(task)

        return make_failure_result(task)


class SequenceRuntime:
    def __init__(
        self,
        results: list[bool],
        *,
        events: list[str] | None = None,
    ) -> None:
        self.results = list(results)
        self.calls: list[Task] = []
        self.capability_calls: list[
            frozenset[str] | None
        ] = []
        self.events = events

    def run(
        self,
        task: Task,
        *,
        allowed_capabilities: Iterable[str] | None = None,
    ) -> AgentLoopResult:
        self.calls.append(task)

        if allowed_capabilities is None:
            self.capability_calls.append(None)
        else:
            self.capability_calls.append(
                frozenset(allowed_capabilities)
            )

        if self.events is not None:
            self.events.append("execute")

        if not self.results:
            raise AssertionError(
                "SequenceRuntime has no result left"
            )

        succeeded = self.results.pop(0)

        if succeeded:
            return make_success_result(task)

        return make_failure_result(task)


class FakePublisher:
    def __init__(
        self,
        *,
        should_fail: bool = False,
        events: list[str] | None = None,
    ) -> None:
        self.should_fail = should_fail
        self.calls: list[
            tuple[ObservedTask, AgentLoopResult]
        ] = []
        self.events = events

    def publish(
        self,
        observed: ObservedTask,
        result: AgentLoopResult,
    ) -> object:
        self.calls.append(
            (
                observed,
                result,
            )
        )

        if self.events is not None:
            self.events.append("publish")

        if self.should_fail:
            raise RuntimeError(
                "publisher failed"
            )

        return object()


class FakeQualifier:
    def __init__(
        self,
        decisions: list[QualificationDecision],
    ) -> None:
        self.decisions = list(decisions)
        self.calls: list[ObservedTask] = []

    def qualify(
        self,
        observed: ObservedTask,
    ) -> QualificationResult:
        self.calls.append(observed)

        if not self.decisions:
            raise AssertionError(
                "FakeQualifier has no decision left"
            )

        decision = self.decisions.pop(0)

        if decision == QualificationDecision.ACCEPT:
            return QualificationResult(
                decision=decision,
                task=observed.task,
                reason="accepted",
                capability=QualificationCapability.CALCULATOR,
            )

        return QualificationResult(
            decision=decision,
            task=None,
            reason=decision.value,
        )


def test_autonomous_agent_polls_task_source() -> None:
    task = make_observed_task()

    source = FakeTaskSource(
        tasks=(task,),
    )

    runtime = FakeRuntime()

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
    )

    run = agent.run_once()

    assert source.poll_count == 1
    assert run.discovered == (task,)


def test_autonomous_agent_executes_accepted_task() -> None:
    task = make_observed_task()

    source = FakeTaskSource(
        tasks=(task,),
    )

    runtime = FakeRuntime()

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        qualifier=qualifier,
    )

    run = agent.run_once()

    assert len(runtime.calls) == 1
    assert runtime.calls[0] is task.task

    assert runtime.capability_calls == [
        frozenset({"calculator"}),
    ]

    assert len(run.results) == 1
    assert run.results[0].result.succeeded is True


def test_autonomous_agent_acknowledges_successful_task() -> None:
    task = make_observed_task(
        message_id=101,
    )

    source = FakeTaskSource(
        tasks=(task,),
    )

    runtime = FakeRuntime()

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        qualifier=qualifier,
    )

    agent.run_once()

    assert source.processed == [101]


def test_autonomous_agent_does_not_acknowledge_failed_execution() -> None:
    task = make_observed_task(
        message_id=102,
    )

    source = FakeTaskSource(
        tasks=(task,),
    )

    runtime = FakeRuntime(
        succeeded=False,
    )

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        qualifier=qualifier,
    )

    run = agent.run_once()

    assert len(run.results) == 1
    assert run.results[0].result.succeeded is False
    assert source.processed == []

    assert runtime.capability_calls == [
        frozenset({"calculator"}),
    ]


def test_autonomous_agent_publishes_successful_result() -> None:
    task = make_observed_task(
        message_id=103,
    )

    source = FakeTaskSource(
        tasks=(task,),
    )

    runtime = FakeRuntime()
    publisher = FakePublisher()

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        publisher=publisher,
        qualifier=qualifier,
    )

    agent.run_once()

    assert len(publisher.calls) == 1

    observed, result = publisher.calls[0]

    assert observed is task
    assert result.result.succeeded is True
    assert source.processed == [103]


def test_autonomous_agent_does_not_acknowledge_when_publisher_fails() -> None:
    task = make_observed_task(
        message_id=104,
    )

    source = FlakyTaskSource(
        task=task,
    )

    runtime = FakeRuntime()

    publisher = FakePublisher(
        should_fail=True,
    )

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        publisher=publisher,
        qualifier=qualifier,
    )

    try:
        agent.run_once()
    except RuntimeError as exc:
        assert str(exc) == "publisher failed"
    else:
        raise AssertionError(
            "expected publisher failure"
        )

    assert publisher.calls
    assert source.processed == []


def test_autonomous_agent_retries_failed_execution() -> None:
    task = make_observed_task(
        message_id=105,
    )

    source = FlakyTaskSource(
        task=task,
    )

    runtime = SequenceRuntime(
        results=[
            False,
            True,
        ],
    )

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        qualifier=qualifier,
    )

    first_run = agent.run_once()

    assert len(first_run.results) == 1
    assert first_run.results[0].result.succeeded is False
    assert source.processed == []

    second_run = agent.run_once()

    assert len(second_run.results) == 1
    assert second_run.results[0].result.succeeded is True

    assert source.processed == [105]
    assert len(runtime.calls) == 2

    assert runtime.capability_calls == [
        frozenset({"calculator"}),
        frozenset({"calculator"}),
    ]


def test_autonomous_agent_retries_after_publisher_failure() -> None:
    task = make_observed_task(
        message_id=106,
    )

    source = FlakyTaskSource(
        task=task,
    )

    runtime = FakeRuntime()

    failing_publisher = FakePublisher(
        should_fail=True,
    )

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        publisher=failing_publisher,
        qualifier=qualifier,
    )

    try:
        agent.run_once()
    except RuntimeError as exc:
        assert str(exc) == "publisher failed"
    else:
        raise AssertionError(
            "expected publisher failure"
        )

    assert source.processed == []

    successful_publisher = FakePublisher()

    agent.publisher = successful_publisher

    second_run = agent.run_once()

    assert len(second_run.results) == 1
    assert second_run.results[0].result.succeeded is True

    assert successful_publisher.calls
    assert source.processed == [106]
    assert len(runtime.calls) == 2

    assert runtime.capability_calls == [
        frozenset({"calculator"}),
        frozenset({"calculator"}),
    ]


def test_autonomous_agent_rejects_task_without_execution() -> None:
    task = make_observed_task(
        message_id=107,
    )

    source = FakeTaskSource(
        tasks=(task,),
    )

    runtime = FakeRuntime()

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.REJECT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        qualifier=qualifier,
    )

    run = agent.run_once()

    assert len(run.results) == 0
    assert len(runtime.calls) == 0
    assert (
        run.qualifications[0].decision
        == QualificationDecision.REJECT
    )
    assert source.processed == [107]
    assert runtime.capability_calls == []


def test_autonomous_agent_ignores_task_without_execution() -> None:
    task = make_observed_task(
        message_id=108,
    )

    source = FakeTaskSource(
        tasks=(task,),
    )

    runtime = FakeRuntime()

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.IGNORE,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        qualifier=qualifier,
    )

    run = agent.run_once()

    assert len(run.results) == 0
    assert len(runtime.calls) == 0
    assert (
        run.qualifications[0].decision
        == QualificationDecision.IGNORE
    )
    assert source.processed == [108]
    assert runtime.capability_calls == []


def test_autonomous_agent_handles_mixed_tasks() -> None:
    accepted = make_observed_task(
        message_id=109,
    )

    rejected = make_observed_task(
        message_id=110,
    )

    ignored = make_observed_task(
        message_id=111,
    )

    source = FakeTaskSource(
        tasks=(
            accepted,
            rejected,
            ignored,
        ),
    )

    runtime = FakeRuntime()

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
            QualificationDecision.REJECT,
            QualificationDecision.IGNORE,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        qualifier=qualifier,
    )

    run = agent.run_once()

    assert len(run.results) == 1
    assert runtime.calls == [accepted.task]

    assert runtime.capability_calls == [
        frozenset({"calculator"}),
    ]

    assert source.processed == [
        109,
        110,
        111,
    ]

    assert run.accepted_count == 1
    assert run.rejected_count == 1
    assert run.ignored_count == 1


def test_autonomous_agent_preserves_task_order() -> None:
    tasks = tuple(
        make_observed_task(
            message_id=message_id,
        )
        for message_id in (
            120,
            121,
            122,
        )
    )

    source = FakeTaskSource(
        tasks=tasks,
    )

    runtime = FakeRuntime()

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
            QualificationDecision.ACCEPT,
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        qualifier=qualifier,
    )

    run = agent.run_once()

    assert [
        result.task_id
        for result in run.results
    ] == [
        tasks[0].task.id,
        tasks[1].task.id,
        tasks[2].task.id,
    ]

    assert [
        task.message_id
        for task in qualifier.calls
    ] == [
        120,
        121,
        122,
    ]

    assert runtime.capability_calls == [
        frozenset({"calculator"}),
        frozenset({"calculator"}),
        frozenset({"calculator"}),
    ]


def test_autonomous_agent_publishes_only_successful_results() -> None:
    successful = make_observed_task(
        message_id=130,
    )

    failed = make_observed_task(
        message_id=131,
    )

    source = FakeTaskSource(
        tasks=(
            successful,
            failed,
        ),
    )

    runtime = SequenceRuntime(
        results=[
            True,
            False,
        ],
    )

    publisher = FakePublisher()

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        publisher=publisher,
        qualifier=qualifier,
    )

    run = agent.run_once()

    assert len(run.results) == 2
    assert len(publisher.calls) == 1

    assert (
        publisher.calls[0][0].message_id
        == 130
    )

    assert source.processed == [130]

    assert runtime.capability_calls == [
        frozenset({"calculator"}),
        frozenset({"calculator"}),
    ]


def test_autonomous_agent_does_not_acknowledge_failed_publisher() -> None:
    task = make_observed_task(
        message_id=140,
    )

    source = FlakyTaskSource(
        task=task,
    )

    runtime = FakeRuntime()

    publisher = FakePublisher(
        should_fail=True,
    )

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        publisher=publisher,
        qualifier=qualifier,
    )

    try:
        agent.run_once()
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "expected publisher failure"
        )

    assert source.processed == []


def test_autonomous_agent_acknowledges_only_after_publish() -> None:
    events: list[str] = []

    task = make_observed_task(
        message_id=150,
    )

    source = FakeTaskSource(
        tasks=(task,),
        events=events,
    )

    runtime = FakeRuntime(
        events=events,
    )

    publisher = FakePublisher(
        events=events,
    )

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        publisher=publisher,
        qualifier=qualifier,
    )

    agent.run_once()

    assert events == [
        "poll",
        "execute",
        "publish",
        "ack:150",
    ]


def test_autonomous_agent_does_not_ack_before_publisher_failure() -> None:
    events: list[str] = []

    task = make_observed_task(
        message_id=151,
    )

    source = FlakyTaskSource(
        task=task,
        events=events,
    )

    runtime = FakeRuntime(
        events=events,
    )

    publisher = FakePublisher(
        should_fail=True,
        events=events,
    )

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        publisher=publisher,
        qualifier=qualifier,
    )

    try:
        agent.run_once()
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "expected publisher failure"
        )

    assert events == [
        "poll",
        "execute",
        "publish",
    ]

    assert "ack:151" not in events
    assert source.processed == []


def test_autonomous_agent_retries_after_restart() -> None:
    task = make_observed_task(
        message_id=160,
    )

    source = FlakyTaskSource(
        task=task,
    )

    failing_runtime = FakeRuntime(
        succeeded=False,
    )

    first_qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
        ],
    )

    first_agent = AutonomousAgent(
        task_source=source,
        runtime=failing_runtime,
        qualifier=first_qualifier,
    )

    first_run = first_agent.run_once()

    assert first_run.results[0].result.succeeded is False
    assert source.processed == []

    successful_runtime = FakeRuntime(
        succeeded=True,
    )

    second_qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
        ],
    )

    second_agent = AutonomousAgent(
        task_source=source,
        runtime=successful_runtime,
        qualifier=second_qualifier,
    )

    second_run = second_agent.run_once()

    assert second_run.results[0].result.succeeded is True
    assert source.processed == [160]

    assert len(failing_runtime.calls) == 1
    assert len(successful_runtime.calls) == 1

    assert failing_runtime.capability_calls == [
        frozenset({"calculator"}),
    ]

    assert successful_runtime.capability_calls == [
        frozenset({"calculator"}),
    ]


def test_autonomous_agent_mixed_success_and_failure_acknowledges_only_success() -> None:
    successful = make_observed_task(
        message_id=170,
    )

    failed = make_observed_task(
        message_id=171,
    )

    source = FakeTaskSource(
        tasks=(
            successful,
            failed,
        ),
    )

    runtime = SequenceRuntime(
        results=[
            True,
            False,
        ],
    )

    qualifier = FakeQualifier(
        decisions=[
            QualificationDecision.ACCEPT,
            QualificationDecision.ACCEPT,
        ],
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
        qualifier=qualifier,
    )

    run = agent.run_once()

    assert len(run.results) == 2

    assert run.results[0].result.succeeded is True
    assert run.results[1].result.succeeded is False

    assert source.processed == [170]

    assert runtime.capability_calls == [
        frozenset({"calculator"}),
        frozenset({"calculator"}),
    ]


def test_autonomous_agent_run_task_executes_explicit_task() -> None:
    source = FakeTaskSource(
        tasks=(),
    )

    runtime = FakeRuntime()

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
    )

    task = Task(
        description="Calculate 10 + 5",
    )

    result = agent.run_task(task)

    assert result.result.succeeded is True
    assert runtime.calls == [task]
    assert runtime.capability_calls == [None]
    assert source.poll_count == 0


def test_autonomous_agent_run_task_rejects_invalid_task() -> None:
    source = FakeTaskSource(
        tasks=(),
    )

    runtime = FakeRuntime()

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
    )

    try:
        agent.run_task("not a task")  # type: ignore[arg-type]
    except TypeError as exc:
        assert str(exc) == "task must be a Task"
    else:
        raise AssertionError(
            "expected TypeError"
        )


def test_autonomous_agent_requires_poll_method() -> None:
    class InvalidSource:
        pass

    runtime = FakeRuntime()

    try:
        AutonomousAgent(
            task_source=InvalidSource(),  # type: ignore[arg-type]
            runtime=runtime,
        )
    except TypeError as exc:
        assert str(exc) == (
            "task_source must provide a poll() method"
        )
    else:
        raise AssertionError(
            "expected TypeError"
        )


def test_autonomous_agent_requires_runtime_run_method() -> None:
    source = FakeTaskSource(
        tasks=(),
    )

    class InvalidRuntime:
        pass

    try:
        AutonomousAgent(
            task_source=source,
            runtime=InvalidRuntime(),  # type: ignore[arg-type]
        )
    except TypeError as exc:
        assert str(exc) == (
            "runtime must provide a run() method"
        )
    else:
        raise AssertionError(
            "expected TypeError"
        )


def test_autonomous_agent_requires_publisher_publish_method() -> None:
    source = FakeTaskSource(
        tasks=(),
    )

    runtime = FakeRuntime()

    class InvalidPublisher:
        pass

    try:
        AutonomousAgent(
            task_source=source,
            runtime=runtime,
            publisher=InvalidPublisher(),  # type: ignore[arg-type]
        )
    except TypeError as exc:
        assert str(exc) == (
            "publisher must provide a publish() method"
        )
    else:
        raise AssertionError(
            "expected TypeError"
        )


def test_autonomous_agent_requires_qualifier_qualify_method() -> None:
    source = FakeTaskSource(
        tasks=(),
    )

    runtime = FakeRuntime()

    class InvalidQualifier:
        pass

    try:
        AutonomousAgent(
            task_source=source,
            runtime=runtime,
            qualifier=InvalidQualifier(),  # type: ignore[arg-type]
        )
    except TypeError as exc:
        assert str(exc) == (
            "qualifier must provide a qualify() method"
        )
    else:
        raise AssertionError(
            "expected TypeError"
        )