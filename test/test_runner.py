from __future__ import annotations

from src.agent.autonomous import AutonomousAgent
from src.agent.loop import AgentLoop
from src.agent.runner import AutonomousRunner
from src.agent.task_source import TechnocoreTaskSource
from src.config import Config


def test_runner_create_builds_production_components() -> None:
    config = Config(
        base_url="https://technocore.chat",
        room="lobby",
    )

    runner = AutonomousRunner.create(
        config=config,
        since=123,
    )

    assert isinstance(runner.agent, AutonomousAgent)
    assert isinstance(runner.task_source, TechnocoreTaskSource)
    assert isinstance(runner.loop, AgentLoop)

    assert runner.task_source.room == "lobby"
    assert runner.task_source.since == 123


def test_runner_create_preserves_execution_limits() -> None:
    runner = AutonomousRunner.create(
        since=50,
        max_iterations=7,
        max_retries=2,
    )

    assert runner.loop.max_iterations == 7
    assert runner.loop.max_retries == 2


def test_runner_uses_same_task_source_and_loop() -> None:
    runner = AutonomousRunner.create()

    assert runner.agent.task_source is runner.task_source
    assert runner.agent.runtime is runner.loop