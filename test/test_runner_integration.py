from __future__ import annotations

from src.agent.loop import AgentLoopResult
from src.agent.runner import AutonomousRunner
from src.agent.task_source import TechnocoreTaskSource
from src.tools.base import ToolResult


class FakeObserver:
    """Return deterministic Technocore observation data."""

    def __init__(
        self,
        *,
        output: str,
    ) -> None:
        self.output = output

    def execute(
        self,
        **kwargs,
    ) -> ToolResult:
        """Return a successful observer tool result."""

        return ToolResult(
            success=True,
            output=self.output,
        )


class FakePublisher:
    """Capture published autonomous results."""

    def __init__(self) -> None:
        self.published = []

    def publish(
        self,
        observed,
        result,
    ) -> None:
        self.published.append(
            (
                observed,
                result,
            )
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

    publisher = FakePublisher()

    runner = AutonomousRunner.create(
        since=499,
        task_source=source,
        publisher=publisher,
    )

    try:
        run = runner.run_once()

        assert len(run.discovered) == 1
        assert run.discovered[0].message_id == 500

        assert len(run.results) == 1

        result = run.results[0]

        assert isinstance(
            result,
            AgentLoopResult,
        )

        assert result.result.succeeded
        assert result.result.output == "96"

        assert len(publisher.published) == 1

        published_observed, published_result = (
            publisher.published[0]
        )

        assert published_observed.message_id == 500
        assert published_result is result

    finally:
        runner.close()
