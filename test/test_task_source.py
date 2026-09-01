from __future__ import annotations

from pathlib import Path

from src.agent.checkpoint_store import SQLiteTaskCheckpointStore
from src.agent.task_source import TechnocoreTaskSource
from src.tools.base import ToolResult


class FakeObserver:
    def __init__(
        self,
        output: str,
    ) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def execute(
        self,
        **kwargs: object,
    ) -> ToolResult:
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


def test_task_source_holds_cursor_before_pending_task() -> None:
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

    # Message 300 is the earliest unprocessed actionable message.
    # The cursor must remain immediately before it.
    assert source.since == 299


def test_task_source_retries_unprocessed_task() -> None:
    observer = FakeObserver(
        """
[message 300]
writer: agent
text: Calculate 10 + 5
"""
    )

    source = TechnocoreTaskSource(
        observer=observer,
        since=0,
    )

    first_tasks = source.poll()

    assert len(first_tasks) == 1
    assert first_tasks[0].message_id == 300
    assert source.since == 299

    second_tasks = source.poll()

    assert len(second_tasks) == 1
    assert second_tasks[0].message_id == 300
    assert source.since == 299

    assert observer.calls == [
        {
            "room": "lobby",
            "since": 0,
        },
        {
            "room": "lobby",
            "since": 299,
        },
    ]


def test_task_source_stops_retrying_after_acknowledgement() -> None:
    observer = FakeObserver(
        """
[message 300]
writer: agent
text: Calculate 10 + 5
"""
    )

    source = TechnocoreTaskSource(
        observer=observer,
        since=0,
    )

    tasks = source.poll()

    assert len(tasks) == 1
    assert tasks[0].message_id == 300
    assert source.since == 299

    source.mark_processed(300)

    next_tasks = source.poll()

    assert next_tasks == ()
    assert source.since == 300

    assert observer.calls == [
        {
            "room": "lobby",
            "since": 0,
        },
        {
            "room": "lobby",
            "since": 299,
        },
    ]


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

    # Message 402 is still pending, so the cursor cannot pass it.
    assert source.since == 401


def test_task_source_advances_past_non_actionable_messages() -> None:
    observer = FakeObserver(
        """
[message 500]
writer: agent
text: Hello everyone

[message 501]
writer: agent
text: Still online
"""
    )

    source = TechnocoreTaskSource(
        observer=observer,
        since=400,
    )

    tasks = source.poll()

    assert tasks == ()
    assert source.since == 501


def test_task_source_raises_when_observer_fails() -> None:
    class FailingObserver:
        def execute(
            self,
            **kwargs: object,
        ) -> ToolResult:
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


def test_task_source_loads_persisted_checkpoint(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        store.set_since(
            "technocore:lobby",
            500,
        )

        observer = FakeObserver(
            """
[message 501]
writer: agent
text: Calculate 12 * 8
"""
        )

        source = TechnocoreTaskSource(
            observer=observer,
            room="lobby",
            since=0,
            checkpoint_store=store,
        )

        assert source.since == 500

        tasks = source.poll()

        assert observer.calls == [
            {
                "room": "lobby",
                "since": 500,
            }
        ]

        assert len(tasks) == 1
        assert tasks[0].message_id == 501

        # The task is still pending.
        assert source.since == 500
        assert store.get_since(
            "technocore:lobby",
        ) == 500


def test_task_source_persists_safe_cursor_after_poll(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    observer = FakeObserver(
        """
[message 600]
writer: agent
text: Calculate 10 + 5

[message 605]
writer: agent
text: Just checking in
"""
    )

    with SQLiteTaskCheckpointStore(database) as store:
        source = TechnocoreTaskSource(
            observer=observer,
            room="lobby",
            since=500,
            checkpoint_store=store,
        )

        tasks = source.poll()

        assert len(tasks) == 1
        assert tasks[0].message_id == 600

        # Message 600 is pending, so 599 is the highest safe cursor.
        assert source.since == 599

        assert store.get_since(
            "technocore:lobby",
        ) == 599


def test_task_source_persists_processed_message(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        observer = FakeObserver(
            """
[message 700]
writer: agent
text: Calculate 12 * 8
"""
        )

        source = TechnocoreTaskSource(
            observer=observer,
            room="lobby",
            since=699,
            checkpoint_store=store,
        )

        tasks = source.poll()

        assert len(tasks) == 1
        assert tasks[0].message_id == 700
        assert source.since == 699

        source.mark_processed(700)

        assert store.is_processed(
            "technocore:lobby",
            700,
        )


def test_task_source_skips_already_processed_messages(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        assert store.mark_processed(
            "technocore:lobby",
            700,
        )

        observer = FakeObserver(
            """
[message 700]
writer: agent
text: Calculate 12 * 8

[message 701]
writer: agent
text: Calculate 10 + 5
"""
        )

        source = TechnocoreTaskSource(
            observer=observer,
            room="lobby",
            since=699,
            checkpoint_store=store,
        )

        tasks = source.poll()

        assert len(tasks) == 1
        assert tasks[0].message_id == 701

        # 700 is already processed, so 701 becomes the pending barrier.
        assert source.since == 700


def test_task_source_persists_across_source_instances(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    first_observer = FakeObserver(
        """
[message 800]
writer: agent
text: Calculate 12 * 8
"""
    )

    with SQLiteTaskCheckpointStore(database) as store:
        first_source = TechnocoreTaskSource(
            observer=first_observer,
            room="lobby",
            since=0,
            checkpoint_store=store,
        )

        first_tasks = first_source.poll()

        assert len(first_tasks) == 1
        assert first_tasks[0].message_id == 800

        # Simulate successful execution and acknowledgement.
        first_source.mark_processed(800)

        assert store.is_processed(
            "technocore:lobby",
            800,
        )

    second_observer = FakeObserver(
        """
[message 800]
writer: agent
text: Calculate 12 * 8

[message 801]
writer: agent
text: Calculate 10 + 5
"""
    )

    with SQLiteTaskCheckpointStore(database) as store:
        second_source = TechnocoreTaskSource(
            observer=second_observer,
            room="lobby",
            since=0,
            checkpoint_store=store,
        )

        # The first source could only safely persist 799 before ACK.
        assert second_source.since == 799

        second_tasks = second_source.poll()

        assert second_observer.calls == [
            {
                "room": "lobby",
                "since": 799,
            }
        ]

        # 800 is already processed, so only 801 is returned.
        assert len(second_tasks) == 1
        assert second_tasks[0].message_id == 801

        assert second_source.since == 800


def test_task_source_retries_after_source_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    first_observer = FakeObserver(
        """
[message 900]
writer: agent
text: Calculate 10 + 5
"""
    )

    with SQLiteTaskCheckpointStore(database) as store:
        first_source = TechnocoreTaskSource(
            observer=first_observer,
            room="lobby",
            since=0,
            checkpoint_store=store,
        )

        tasks = first_source.poll()

        assert len(tasks) == 1
        assert tasks[0].message_id == 900

        # No acknowledgement: simulate runtime failure or crash.
        assert first_source.since == 899

        assert store.get_since(
            "technocore:lobby",
        ) == 899

    second_observer = FakeObserver(
        """
[message 900]
writer: agent
text: Calculate 10 + 5
"""
    )

    with SQLiteTaskCheckpointStore(database) as store:
        second_source = TechnocoreTaskSource(
            observer=second_observer,
            room="lobby",
            since=0,
            checkpoint_store=store,
        )

        assert second_source.since == 899

        retried_tasks = second_source.poll()

        assert len(retried_tasks) == 1
        assert retried_tasks[0].message_id == 900


def test_task_source_handles_multiple_pending_tasks(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    observer = FakeObserver(
        """
[message 1000]
writer: agent-a
text: Calculate 10 + 5

[message 1001]
writer: agent-b
text: Calculate 20 + 5

[message 1002]
writer: agent-c
text: Just checking in
"""
    )

    with SQLiteTaskCheckpointStore(database) as store:
        source = TechnocoreTaskSource(
            observer=observer,
            room="lobby",
            since=999,
            checkpoint_store=store,
        )

        tasks = source.poll()

        assert len(tasks) == 2
        assert [
            task.message_id
            for task in tasks
        ] == [1000, 1001]

        # The earliest pending task is 1000.
        assert source.since == 999

        # Acknowledge only the first task.
        source.mark_processed(1000)

        next_tasks = source.poll()

        assert len(next_tasks) == 1
        assert next_tasks[0].message_id == 1001

        # 1001 is still pending.
        assert source.since == 1000

        # Acknowledge the second task.
        source.mark_processed(1001)

        final_tasks = source.poll()

        assert final_tasks == ()
        assert source.since == 1002


def test_task_source_rejects_invalid_checkpoint_source(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        observer = FakeObserver("")

        try:
            TechnocoreTaskSource(
                observer=observer,
                checkpoint_store=store,
                checkpoint_source="",
            )
        except ValueError as exc:
            assert str(exc) == (
                "checkpoint_source cannot be empty"
            )
        else:
            raise AssertionError(
                "expected ValueError"
            )


def test_task_source_rejects_invalid_checkpoint_source_type(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        observer = FakeObserver("")

        try:
            TechnocoreTaskSource(
                observer=observer,
                checkpoint_store=store,
                checkpoint_source=123,  # type: ignore[arg-type]
            )
        except TypeError as exc:
            assert str(exc) == (
                "checkpoint_source must be a string or None"
            )
        else:
            raise AssertionError(
                "expected TypeError"
            )


def test_task_source_mark_processed_without_checkpoint_store() -> None:
    observer = FakeObserver(
        """
[message 1100]
writer: agent
text: Calculate 10 + 5
"""
    )

    source = TechnocoreTaskSource(
        observer=observer,
        since=1099,
    )

    tasks = source.poll()

    assert len(tasks) == 1
    assert tasks[0].message_id == 1100

    source.mark_processed(1100)

    next_tasks = source.poll()

    assert next_tasks == ()
    assert source.since == 1100


def test_task_source_rejects_invalid_processed_message_id() -> None:
    observer = FakeObserver("")

    source = TechnocoreTaskSource(
        observer=observer,
    )

    try:
        source.mark_processed(-1)
    except ValueError as exc:
        assert str(exc) == (
            "message_id cannot be negative"
        )
    else:
        raise AssertionError(
            "expected ValueError"
        )


def test_task_source_rejects_invalid_processed_message_id_type() -> None:
    observer = FakeObserver("")

    source = TechnocoreTaskSource(
        observer=observer,
    )

    try:
        source.mark_processed("100")  # type: ignore[arg-type]
    except TypeError as exc:
        assert str(exc) == (
            "message_id must be an integer"
        )
    else:
        raise AssertionError(
            "expected TypeError"
        )