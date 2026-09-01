from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.agent.autonomous import AutonomousRun
from src.agent.daemon import AutonomousDaemon, DaemonCycle
from src.agent.result import ExecutionResult
from src.agent.task import Task, TaskStatus


@dataclass
class FakeRunner:
    """Minimal runner used to test daemon lifecycle behavior."""

    runs: int = 0
    closed: bool = False
    should_fail: bool = False

    def run_once(self) -> AutonomousRun:
        self.runs += 1

        if self.should_fail:
            raise RuntimeError(
                "temporary runner failure"
            )

        task = Task(
            description="Calculate 10 + 5"
        )

        result = ExecutionResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            executed_steps=1,
            output=15,
        )

        from src.agent.loop import AgentLoopResult
        from src.agent.decision import AutonomyAction

        loop_result = AgentLoopResult(
            task_id=task.id,
            result=result,
            iterations=1,
            action=AutonomyAction.COMPLETE,
        )

        from src.agent.task_source import ObservedTask

        observed = ObservedTask(
            task=task,
            message_id=100 + self.runs,
            writer="test-agent",
            text=task.description,
        )

        return AutonomousRun(
            discovered=(observed,),
            results=(loop_result,),
        )

    def close(self) -> None:
        self.closed = True


class EmptyRunner:
    """Runner returning an empty autonomous cycle."""

    def __init__(self) -> None:
        self.runs = 0
        self.closed = False

    def run_once(self) -> AutonomousRun:
        self.runs += 1

        return AutonomousRun(
            discovered=(),
            results=(),
        )

    def close(self) -> None:
        self.closed = True


def test_daemon_rejects_invalid_runner() -> None:
    with pytest.raises(TypeError):
        AutonomousDaemon(
            runner=object(),  # type: ignore[arg-type]
        )


def test_daemon_rejects_negative_interval() -> None:
    runner = FakeRunner()

    with pytest.raises(ValueError):
        AutonomousDaemon(
            runner=runner,  # type: ignore[arg-type]
            interval=-1,
        )


def test_daemon_rejects_invalid_interval() -> None:
    runner = FakeRunner()

    with pytest.raises(TypeError):
        AutonomousDaemon(
            runner=runner,  # type: ignore[arg-type]
            interval="30",  # type: ignore[arg-type]
        )


def test_daemon_run_cycle_counts_completed_tasks() -> None:
    runner = FakeRunner()

    daemon = AutonomousDaemon(
        runner=runner,  # type: ignore[arg-type]
        interval=0,
    )

    cycle = daemon.run_cycle()

    assert cycle == DaemonCycle(
        discovered=1,
        completed=1,
        failed=0,
        error=None,
    )

    assert runner.runs == 1


def test_daemon_run_cycle_captures_runner_errors() -> None:
    runner = FakeRunner(
        should_fail=True
    )

    daemon = AutonomousDaemon(
        runner=runner,  # type: ignore[arg-type]
        interval=0,
    )

    cycle = daemon.run_cycle()

    assert cycle.discovered == 0
    assert cycle.completed == 0
    assert cycle.failed == 0
    assert cycle.error == (
        "temporary runner failure"
    )


def test_daemon_runs_requested_number_of_cycles() -> None:
    runner = EmptyRunner()
    sleeps: list[float] = []

    daemon = AutonomousDaemon(
        runner=runner,  # type: ignore[arg-type]
        interval=12,
        sleep=sleeps.append,
    )

    daemon.run_forever(
        max_cycles=3
    )

    assert runner.runs == 3
    assert sleeps == [12.0, 12.0]
    assert daemon.running is False


def test_daemon_stops_from_cycle_callback() -> None:
    runner = EmptyRunner()

    daemon = AutonomousDaemon(
        runner=runner,  # type: ignore[arg-type]
        interval=0,
        sleep=lambda _: None,
    )

    def stop_after_first(
        cycle: DaemonCycle,
    ) -> None:
        assert cycle.discovered == 0
        daemon.stop()

    daemon.run_forever(
        on_cycle=stop_after_first
    )

    assert runner.runs == 1
    assert daemon.running is False


def test_daemon_stop_is_idempotent() -> None:
    runner = EmptyRunner()

    daemon = AutonomousDaemon(
        runner=runner,  # type: ignore[arg-type]
    )

    daemon.stop()
    daemon.stop()

    assert daemon.running is False


def test_daemon_rejects_invalid_max_cycles() -> None:
    runner = EmptyRunner()

    daemon = AutonomousDaemon(
        runner=runner,  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError):
        daemon.run_forever(
            max_cycles="3",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError):
        daemon.run_forever(
            max_cycles=0
        )


def test_daemon_context_manager_closes_runner() -> None:
    runner = EmptyRunner()

    with AutonomousDaemon(
        runner=runner,  # type: ignore[arg-type]
    ) as daemon:
        assert daemon.running is False

    assert runner.closed is True


def test_daemon_close_stops_and_closes_runner() -> None:
    runner = EmptyRunner()

    daemon = AutonomousDaemon(
        runner=runner,  # type: ignore[arg-type]
    )

    daemon.close()

    assert daemon.running is False
    assert runner.closed is True