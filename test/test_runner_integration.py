from __future__ import annotations

from dataclasses import dataclass

from src.agent.runner import AutonomousRunner
from src.agent.task import TaskStatus
from src.agent.task_source import TechnocoreTaskSource


@dataclass(frozen=True, slots=True)
class FakeObservationResult:
    """Minimal result object returned by the fake observer."""

    success: bool
    output: str | None = None
    error: str | None = None


class FakeObserver:
    """Fake Technocore observer for integration testing."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[str, int]] = []

    def execute(
        self,
        *,
        room: str,
        since: int,
    ) -> FakeObservationResult:
        self.calls.append(
            (room, since)
        )

        return FakeObservationResult(
            success=True,
            output=self.output,
        )


def test_autonomous_runner_executes_discovered_calculation() -> None:
    observer = FakeObserver(
        output=(
            "[message 500]\n"
            "writer: test-agent\n"
            "text: Calculate 12 * 8\n"
        ),
    )

    source = TechnocoreTaskSource(
        observer=observer,
        room="lobby",
        since=499,
    )

    runner = AutonomousRunner.create(
        since=499,
        task_source=source,
    )

    run = runner.run_once()

    assert observer.calls == [
        ("lobby", 499),
    ]

    assert len(run.discovered) == 1

    assert run.discovered[0].message_id == 500

    assert run.discovered[0].text == (
        "Calculate 12 * 8"
    )

    assert len(run.results) == 1

    result = run.results[0]

    assert result.status == TaskStatus.COMPLETED
    assert result.succeeded
    assert result.output == "96"

    assert source.since == 500