from __future__ import annotations

from src.agent.autonomous import AutonomousAgent
from src.agent.loop import AgentLoop
from src.agent.task import Task
from src.agent.task_source import ObservedTask


class CalculatorTaskSource:
    """Task source that provides one calculator task."""

    def __init__(self) -> None:
        self.poll_count = 0

        task = Task(
            description="Calculate 12 * 8",
        )

        self.observed = ObservedTask(
            task=task,
            message_id=500,
            writer="test-agent",
            text="Calculate 12 * 8",
        )

    def poll(self) -> tuple[ObservedTask, ...]:
        self.poll_count += 1
        return (self.observed,)


def test_autonomous_agent_executes_discovered_calculator_task() -> None:
    source = CalculatorTaskSource()

    agent = AutonomousAgent(
        task_source=source,
    )

    assert isinstance(
        agent.runtime,
        AgentLoop,
    )

    run = agent.run_once()

    assert source.poll_count == 1

    assert len(run.discovered) == 1

    assert run.discovered[0] is source.observed

    assert len(run.results) == 1

    result = run.results[0]

    assert result.task_id == source.observed.task.id

    assert result.result.succeeded

    assert result.result.output == "96"

    assert result.completed

    assert source.observed.task.status.value == "completed"