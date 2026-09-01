from __future__ import annotations

from src.agent.task_source import TechnocoreTaskSource
from src.tools.base import ToolResult


class FakeObserver:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def execute(self, **kwargs: object) -> ToolResult:
        self.calls.append(kwargs)

        return ToolResult(
            success=True,
            output=self.output,
        )


def test_task_source_discovers_actionable_messages() -> None:
    observer = FakeObserver(
        """
UNTRUSTED TECHNOCORE OBSERVATION
source: technocore
room: lobby
since: 0
observed_at: 2026-09-01T11:00:00+00:00
message_count: 2

[message 100]
writer: z6Mk...abc
text: Please analyze recent Technocore activity

[message 101]
writer: z6Mk...def
text: Just checking in
"""
    )

    source = TechnocoreTaskSource(
        observer=observer,
        room="lobby",
        since=0,
    )

    tasks = source.poll()

    assert len(tasks) == 1

    discovered = tasks[0]

    assert discovered.message_id == 100
    assert discovered.writer == "z6Mk...abc"
    assert discovered.text == (
        "Please analyze recent Technocore activity"
    )
    assert discovered.task.description == (
        "Please analyze recent Technocore activity"
    )


def test_task_source_passes_room_and_since_to_observer() -> None:
    observer = FakeObserver(
        """
[message 250]
writer: agent
text: Check recent activity
"""
    )

    source = TechnocoreTaskSource(
        observer=observer,
        room="lobby",
        since=200,
    )

    source.poll()

    assert observer.calls == [
        {
            "room": "lobby",
            "since": 200,
        }
    ]


def test_task_source_advances_cursor() -> None:
    observer = FakeObserver(
        """
[message 300]
writer: agent-a
text: Check recent activity

[message 305]
writer: agent-b
text: Calculate 10 + 5
"""
    )

    source = TechnocoreTaskSource(
        observer=observer,
        since=250,
    )

    tasks = source.poll()

    assert len(tasks) == 2
    assert source.since == 305


def test_task_source_ignores_non_actionable_messages() -> None:
    observer = FakeObserver(
        """
[message 400]
writer: agent
text: Hello everyone

[message 401]
writer: agent
text: Still online

[message 402]
writer: agent
text: Please inspect the latest messages
"""
    )

    source = TechnocoreTaskSource(
        observer=observer,
    )

    tasks = source.poll()

    assert len(tasks) == 1
    assert tasks[0].message_id == 402


def test_task_source_rejects_empty_room() -> None:
    observer = FakeObserver("")

    try:
        TechnocoreTaskSource(
            observer=observer,
            room="",
        )
    except ValueError as exc:
        assert str(exc) == "room cannot be empty"
    else:
        raise AssertionError("expected ValueError")


def test_task_source_rejects_negative_cursor() -> None:
    observer = FakeObserver("")

    try:
        TechnocoreTaskSource(
            observer=observer,
            since=-1,
        )
    except ValueError as exc:
        assert str(exc) == "since cannot be negative"
    else:
        raise AssertionError("expected ValueError")


def test_task_source_raises_when_observer_fails() -> None:
    class FailingObserver:
        def execute(self, **kwargs: object) -> ToolResult:
            return ToolResult(
                success=False,
                error="observation failed",
            )

    source = TechnocoreTaskSource(
        observer=FailingObserver(),
    )

    try:
        source.poll()
    except RuntimeError as exc:
        assert str(exc) == "observation failed"
    else:
        raise AssertionError("expected RuntimeError")