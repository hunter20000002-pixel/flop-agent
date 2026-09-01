from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.checkpoint_store import SQLiteTaskCheckpointStore
from src.agent.task import Task
from src.tools.technocore import TechnocoreObserverTool


@dataclass(frozen=True, slots=True)
class ObservedTask:
    """A task candidate discovered from an external observation."""

    task: Task
    message_id: int
    writer: str
    text: str


@dataclass(frozen=True, slots=True)
class ObservedTaskData:
    """Internal representation of a parsed Technocore message."""

    message_id: int
    writer: str
    text: str


class TechnocoreTaskSource:
    """Discovers executable tasks from Technocore observations.

    The source uses at-least-once delivery semantics.

    The observation cursor is never advanced past an unprocessed
    actionable message. This ensures that tasks which fail during
    execution or publishing remain discoverable on the next poll.

    Processed message IDs are persisted when a checkpoint store is
    configured. Without a checkpoint store, processed IDs are retained
    in memory for the lifetime of this source instance.
    """

    def __init__(
        self,
        *,
        observer: TechnocoreObserverTool | None = None,
        room: str = "lobby",
        since: int = 0,
        checkpoint_store: SQLiteTaskCheckpointStore | None = None,
        checkpoint_source: str | None = None,
    ) -> None:
        if not isinstance(room, str):
            raise TypeError(
                "room must be a string"
            )

        if not room.strip():
            raise ValueError(
                "room cannot be empty"
            )

        if not isinstance(since, int):
            raise TypeError(
                "since must be an integer"
            )

        if since < 0:
            raise ValueError(
                "since cannot be negative"
            )

        if checkpoint_source is not None:
            if not isinstance(checkpoint_source, str):
                raise TypeError(
                    "checkpoint_source must be a string or None"
                )

            if not checkpoint_source.strip():
                raise ValueError(
                    "checkpoint_source cannot be empty"
                )

        self.observer = (
            observer
            or TechnocoreObserverTool()
        )

        self.room = room
        self.checkpoint_store = checkpoint_store

        self.checkpoint_source = (
            checkpoint_source
            if checkpoint_source is not None
            else f"technocore:{room}"
        )

        self._processed_message_ids: set[int] = set()

        if checkpoint_store is None:
            self.since = since
        else:
            self.since = checkpoint_store.get_since(
                self.checkpoint_source,
                default=since,
            )

    def poll(self) -> tuple[ObservedTask, ...]:
        """
        Observe Technocore and return newly discovered actionable tasks.

        The cursor only advances past messages that are safe to forget:
        non-actionable messages and already-processed actionable messages.

        An unprocessed actionable message acts as a delivery barrier.
        The cursor remains immediately before the earliest such message,
        allowing failed tasks to be rediscovered on subsequent polls.
        """

        cursor = self.since

        result = self.observer.execute(
            room=self.room,
            since=cursor,
        )

        if not result.success:
            raise RuntimeError(
                result.error
                or "Technocore observation failed"
            )

        observations = self._parse_observations(
            result.output
        )

        if not observations:
            return ()

        new_observations = tuple(
            sorted(
                (
                    observation
                    for observation in observations
                    if observation.message_id > cursor
                ),
                key=lambda observation: observation.message_id,
            )
        )

        if not new_observations:
            return ()

        actionable: list[ObservedTask] = []
        earliest_pending_id: int | None = None

        for observation in new_observations:
            if not self._is_actionable(
                observation.text
            ):
                continue

            if self._is_processed(
                observation.message_id
            ):
                continue

            if earliest_pending_id is None:
                earliest_pending_id = (
                    observation.message_id
                )

            actionable.append(
                self._to_observed_task(
                    observation
                )
            )

        if earliest_pending_id is None:
            newest_message_id = max(
                observation.message_id
                for observation in new_observations
            )

            self._advance_cursor(
                newest_message_id
            )
        else:
            safe_cursor = (
                earliest_pending_id - 1
            )

            if safe_cursor > self.since:
                self._advance_cursor(
                    safe_cursor
                )

        return tuple(actionable)

    def mark_processed(
        self,
        message_id: int,
    ) -> None:
        """Acknowledge successful processing of a message."""

        if not isinstance(message_id, int):
            raise TypeError(
                "message_id must be an integer"
            )

        if message_id < 0:
            raise ValueError(
                "message_id cannot be negative"
            )

        self._processed_message_ids.add(
            message_id
        )

        if self.checkpoint_store is None:
            return

        self.checkpoint_store.mark_processed(
            self.checkpoint_source,
            message_id,
        )

    def _is_processed(
        self,
        message_id: int,
    ) -> bool:
        """Return whether a message has already been acknowledged."""

        if message_id in self._processed_message_ids:
            return True

        if self.checkpoint_store is None:
            return False

        return self.checkpoint_store.is_processed(
            self.checkpoint_source,
            message_id,
        )

    def _advance_cursor(
        self,
        since: int,
    ) -> None:
        """Advance and persist the observation cursor."""

        if since <= self.since:
            return

        self.since = since

        if self.checkpoint_store is not None:
            self.checkpoint_store.set_since(
                self.checkpoint_source,
                self.since,
            )

    @staticmethod
    def _parse_observations(
        output: Any,
    ) -> tuple[ObservedTaskData, ...]:
        """Parse structured Technocore observation text."""

        if not isinstance(output, str):
            return ()

        observations: list[ObservedTaskData] = []

        current_message_id: int | None = None
        current_writer = ""
        current_text = ""

        for raw_line in output.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            if (
                line.startswith("[message ")
                and line.endswith("]")
            ):
                if current_message_id is not None:
                    observations.append(
                        ObservedTaskData(
                            message_id=current_message_id,
                            writer=current_writer,
                            text=current_text,
                        )
                    )

                raw_id = line[
                    len("[message "):-1
                ].strip()

                try:
                    current_message_id = int(raw_id)
                except ValueError:
                    current_message_id = None

                current_writer = ""
                current_text = ""
                continue

            if current_message_id is None:
                continue

            if line.startswith("writer:"):
                current_writer = line[
                    len("writer:"):
                ].strip()
                continue

            if line.startswith("text:"):
                current_text = line[
                    len("text:"):
                ].strip()
                continue

        if current_message_id is not None:
            observations.append(
                ObservedTaskData(
                    message_id=current_message_id,
                    writer=current_writer,
                    text=current_text,
                )
            )

        return tuple(observations)

    @staticmethod
    def _is_actionable(
        text: str,
    ) -> bool:
        """Return True when an observation contains a task-like request."""

        normalized = text.strip().lower()

        if not normalized:
            return False

        action_prefixes = (
            "please ",
            "can you ",
            "could you ",
            "check ",
            "analyze ",
            "analyse ",
            "inspect ",
            "research ",
            "find ",
            "calculate ",
            "compute ",
            "explain ",
            "summarize ",
            "read ",
            "list ",
            "show ",
        )

        return normalized.startswith(
            action_prefixes
        )

    @staticmethod
    def _to_observed_task(
        observation: ObservedTaskData,
    ) -> ObservedTask:
        """Convert an observation into an agent Task."""

        task = Task(
            description=observation.text,
        )

        return ObservedTask(
            task=task,
            message_id=observation.message_id,
            writer=observation.writer,
            text=observation.text,
        )