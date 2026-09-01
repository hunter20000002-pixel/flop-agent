from __future__ import annotations

from src.agent.autonomous import AutonomousAgent
from src.agent.loop import AgentLoop, AgentLoopResult
from src.agent.decision import AutonomyAction
from src.agent.result import ExecutionResult
from src.agent.task import Task, TaskStatus
from src.agent.task_source import ObservedTask


class FakeTaskSource:
    """Simple source used to test autonomous orchestration."""

    def __init__(
        self,
        discovered: tuple[ObservedTask, ...],
    ) -> None:
        self.discovered = discovered
        self.poll_count = 0

    def poll(self) -> tuple[ObservedTask, ...]:
        self.poll_count += 1
        return self.discovered


class FakeRuntime:
    """Simple autonomous-loop executor used to verify task dispatch."""

    def __init__(self) -> None:
        self.tasks: list[Task] = []

    def run(self, task: Task) -> AgentLoopResult:
        self.tasks.append(task)

        execution_result = ExecutionResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            executed_steps=1,
            output="test output",
        )

        return AgentLoopResult(
            task_id=task.id,
            result=execution_result,
            iterations=1,
            action=AutonomyAction.COMPLETE,
        )


def make_observed_task(
    description: str,
    message_id: int,
) -> ObservedTask:
    """Create a test observation containing a Task."""

    task = Task(
        description=description,
    )

    return ObservedTask(
        task=task,
        message_id=message_id,
        writer="test-agent",
        text=description,
    )


def test_autonomous_agent_runs_discovered_tasks() -> None:
    first = make_observed_task(
        "Calculate 10 + 5",
        100,
    )

    second = make_observed_task(
        "Analyze recent activity",
        101,
    )

    source = FakeTaskSource(
        discovered=(
            first,
            second,
        ),
    )

    runtime = FakeRuntime()

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
    )

    run = agent.run_once()

    assert source.poll_count == 1

    assert run.discovered == (
        first,
        second,
    )

    assert len(run.results) == 2

    assert runtime.tasks == [
        first.task,
        second.task,
    ]

    assert all(
        result.action == AutonomyAction.COMPLETE
        for result in run.results
    )


def test_autonomous_agent_handles_no_discovered_tasks() -> None:
    source = FakeTaskSource(
        discovered=(),
    )

    runtime = FakeRuntime()

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
    )

    run = agent.run_once()

    assert run.discovered == ()
    assert run.results == ()
    assert runtime.tasks == []


def test_autonomous_agent_preserves_discovery_order() -> None:
    first = make_observed_task(
        "First task",
        200,
    )

    second = make_observed_task(
        "Second task",
        201,
    )

    third = make_observed_task(
        "Third task",
        202,
    )

    source = FakeTaskSource(
        discovered=(
            first,
            second,
            third,
        ),
    )

    runtime = FakeRuntime()

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
    )

    run = agent.run_once()

    assert [
        observed.message_id
        for observed in run.discovered
    ] == [
        200,
        201,
        202,
    ]

    assert runtime.tasks == [
        first.task,
        second.task,
        third.task,
    ]


def test_autonomous_agent_can_execute_direct_task() -> None:
    source = FakeTaskSource(
        discovered=(),
    )

    runtime = FakeRuntime()

    agent = AutonomousAgent(
        task_source=source,
        runtime=runtime,
    )

    task = Task(
        description="Direct task",
    )

    result = agent.run_task(task)

    assert isinstance(result, AgentLoopResult)
    assert result.task_id == task.id
    assert result.result.status == TaskStatus.COMPLETED
    assert result.action == AutonomyAction.COMPLETE
    assert runtime.tasks == [task]


def test_autonomous_agent_rejects_invalid_source() -> None:
    runtime = FakeRuntime()

    try:
        AutonomousAgent(
            task_source=object(),
            runtime=runtime,
        )
    except TypeError as exc:
        assert str(exc) == (
            "task_source must provide a poll() method"
        )
    else:
        raise AssertionError("expected TypeError")


def test_autonomous_agent_rejects_invalid_runtime() -> None:
    source = FakeTaskSource(
        discovered=(),
    )

    try:
        AutonomousAgent(
            task_source=source,
            runtime=object(),
        )
    except TypeError as exc:
        assert str(exc) == (
            "runtime must provide a run() method"
        )
    else:
        raise AssertionError("expected TypeError")
    
def test_autonomous_agent_defaults_to_agent_loop() -> None:
    source = FakeTaskSource(
        discovered=(),
    )

    agent = AutonomousAgent(
        task_source=source,
    )

    assert isinstance(
        agent.runtime,
        AgentLoop,
    )