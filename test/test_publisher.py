from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.autonomous import AutonomousAgent
from src.agent.loop import AgentLoopResult
from src.agent.publisher import (
    PublishedMessage,
    TechnocoreResultPublisher,
)
from src.agent.result import ExecutionResult
from src.agent.task import Task, TaskStatus
from src.agent.task_source import ObservedTask
from src.agent.decision import AutonomyAction
from src.config import Config


class FakeRuntime:
    def __init__(
        self,
        *,
        succeeded: bool = True,
    ) -> None:
        self.succeeded = succeeded
        self.received_capabilities = None

    def run(
        self,
        task: Task,
        *,
        allowed_capabilities=None,
    ) -> AgentLoopResult:
        self.received_capabilities = allowed_capabilities

        status = (
            TaskStatus.COMPLETED
            if self.succeeded
            else TaskStatus.FAILED
        )

        execution_result = ExecutionResult(
            task_id=task.id,
            status=status,
            executed_steps=1,
            output=(
                "calculated result"
                if self.succeeded
                else None
            ),
            error=(
                None
                if self.succeeded
                else "execution failed"
            ),
        )

        return AgentLoopResult(
            task_id=task.id,
            result=execution_result,
            iterations=1,
            action=(
                AutonomyAction.COMPLETE
                if self.succeeded
                else AutonomyAction.STOP
            ),
        )


class FakeSource:
    def __init__(
        self,
        observed: ObservedTask,
    ) -> None:
        self.observed = observed
        self.processed: list[int] = []

    def poll(self) -> tuple[ObservedTask, ...]:
        return (self.observed,)

    def mark_processed(
        self,
        message_id: int,
    ) -> None:
        self.processed.append(message_id)


class FakePublisher:
    def __init__(
        self,
        *,
        should_fail: bool = False,
    ) -> None:
        self.should_fail = should_fail
        self.published: list[
            tuple[ObservedTask, AgentLoopResult]
        ] = []

    def publish(
        self,
        observed: ObservedTask,
        result: AgentLoopResult,
    ) -> PublishedMessage:
        if self.should_fail:
            raise RuntimeError(
                "publication failed"
            )

        self.published.append(
            (observed, result)
        )

        return PublishedMessage(
            response_text="test result",
            nonce="123",
            did="did:key:test",
            server_response="ok",
        )


def make_observed_task() -> ObservedTask:
    task = Task(
        description="Calculate 10 + 5",
    )

    return ObservedTask(
        task=task,
        message_id=123,
        writer="agent",
        text="Calculate 10 + 5",
    )


def test_format_result_contains_execution_metadata() -> None:
    observed = make_observed_task()

    result = FakeRuntime().run(
        observed.task
    )

    text = (
        TechnocoreResultPublisher._format_result(
            observed,
            result,
        )
    )

    assert "FLOP Agent autonomous execution result" in text
    assert "source_message: 123" in text
    assert f"task_id: {result.task_id}" in text
    assert "status: completed" in text
    assert "success: true" in text
    assert "output: calculated result" in text


def test_format_result_contains_failure() -> None:
    observed = make_observed_task()

    result = FakeRuntime(
        succeeded=False,
    ).run(
        observed.task
    )

    text = (
        TechnocoreResultPublisher._format_result(
            observed,
            result,
        )
    )

    assert "success: false" in text
    assert "error: execution failed" in text


def test_publisher_accepts_custom_key_file(
    tmp_path: Path,
) -> None:
    config = Config(
        key_file=tmp_path / "identity.json",
    )

    publisher = TechnocoreResultPublisher(
        config=config,
    )

    assert publisher.did is None
    assert publisher.key_file == (
        tmp_path / "identity.json"
    )


def test_publisher_rejects_invalid_config() -> None:
    with pytest.raises(TypeError):
        TechnocoreResultPublisher(
            config="invalid",  # type: ignore[arg-type]
        )


def test_successful_publication_is_acknowledged() -> None:
    observed = make_observed_task()

    source = FakeSource(
        observed
    )

    publisher = FakePublisher()

    agent = AutonomousAgent(
        task_source=source,
        runtime=FakeRuntime(),
        publisher=publisher,
    )

    run = agent.run_once()

    assert len(run.results) == 1
    assert len(publisher.published) == 1
    assert source.processed == [123]


def test_failed_execution_is_not_published() -> None:
    observed = make_observed_task()

    source = FakeSource(
        observed
    )

    publisher = FakePublisher()

    agent = AutonomousAgent(
        task_source=source,
        runtime=FakeRuntime(
            succeeded=False,
        ),
        publisher=publisher,
    )

    run = agent.run_once()

    assert len(run.results) == 1
    assert publisher.published == []
    assert source.processed == []


def test_failed_publication_is_not_acknowledged() -> None:
    observed = make_observed_task()

    source = FakeSource(
        observed
    )

    publisher = FakePublisher(
        should_fail=True,
    )

    agent = AutonomousAgent(
        task_source=source,
        runtime=FakeRuntime(),
        publisher=publisher,
    )

    with pytest.raises(
        RuntimeError,
        match="publication failed",
    ):
        agent.run_once()

    assert source.processed == []
