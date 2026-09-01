from __future__ import annotations

from pathlib import Path

from src.agent.runner import AutonomousRunner
from src.agent.task_source import TechnocoreTaskSource


def test_runner_create_builds_autonomous_agent() -> None:
    runner = AutonomousRunner.create()

    assert runner.agent is not None
    assert runner.task_source is not None
    assert runner.loop is not None
    assert runner.checkpoint_store is None


def test_runner_create_uses_supplied_task_source() -> None:
    source = TechnocoreTaskSource(
        since=123,
    )

    runner = AutonomousRunner.create(
        task_source=source,
    )

    assert runner.task_source is source
    assert runner.checkpoint_store is None


def test_runner_create_can_use_checkpoint_store(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runner.db"

    runner = AutonomousRunner.create(
        checkpoint_path=database,
    )

    try:
        assert runner.checkpoint_store is not None

        assert runner.task_source.checkpoint_store is (
            runner.checkpoint_store
        )

        assert runner.task_source.since == 0
    finally:
        runner.close()


def test_runner_checkpoint_store_persists_between_runners(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runner.db"

    with AutonomousRunner.create(
        checkpoint_path=database,
    ) as first_runner:
        assert first_runner.checkpoint_store is not None

        first_runner.checkpoint_store.set_since(
            "technocore:lobby",
            500,
        )

    with AutonomousRunner.create(
        checkpoint_path=database,
    ) as second_runner:
        assert second_runner.task_source.since == 500


def test_runner_close_is_safe_without_checkpoint_store() -> None:
    runner = AutonomousRunner.create()

    runner.close()
    runner.close()


def test_runner_context_manager_closes_checkpoint_store(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runner.db"

    with AutonomousRunner.create(
        checkpoint_path=database,
    ) as runner:
        checkpoint_store = runner.checkpoint_store

        assert checkpoint_store is not None

    assert checkpoint_store is not None
