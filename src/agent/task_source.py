from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.task import Task
from src.tools.technocore import TechnocoreObserverTool


@dataclass(frozen=True, slots=True)
class ObservedTask:
    """A task candidate discovered from an external observation."""

    task: Task
    message_id: int
    writer: str
    text: str


class TechnocoreTaskSource:
    """Discovers executable tasks from Technocore observations."""

    def __init__(
        self,
        *,
        observer: TechnocoreObserverTool | None = None,
        room: str = "lobby",
        since: int = 0,
    ) -> None:
        if not room.strip():
            raise ValueError("room cannot be empty")

        if since < 0:
            raise ValueError("since cannot be negative")

        self.observer = observer or TechnocoreObserverTool()
        self.room = room
        self.since = since

    def poll(self) -> tuple[ObservedTask, ...]:
        """Observe Technocore and convert actionable messages into tasks."""

        result = self.observer.execute(
            room=self.room,
            since=self.since,
        )

        if not result.success:
            raise RuntimeError(
                result.error or "Technocore observation failed"
            )

        observations = self._parse_observations(
            result.output
        )

        if observations:
            self.since = max(
                observation.message_id
                for observation in observations
            )

        return tuple(
            self._to_observed_task(observation)
            for observation in observations
            if self._is_actionable(observation.text)
        )

    @staticmethod
    def _parse_observations(
        output: Any,
    ) -> tuple[ObservedTaskData, ...]:
        """Parse structured observation text into message records."""

        if not isinstance(output, str):
            return ()

        lines = output.splitlines()

        observations: list[ObservedTaskData] = []

        message_id: int | None = None
        writer = ""
        text = ""

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("[message ") and stripped.endswith("]"):
                if message_id is not None:
                    observations.append(
                        ObservedTaskData(
                            message_id=message_id,
                            writer=writer,
                            text=text,
                        )
                    )

                raw_id = stripped[
                    len("[message "):-1
                ]

                try:
                    message_id = int(raw_id)
                except ValueError:
                    message_id = None

                writer = ""
                text = ""
                continue

            if message_id is None:
                continue

            if stripped.startswith("writer:"):
                writer = stripped[
                    len("writer:"):].strip()
                continue

            if stripped.startswith("text:"):
                text = stripped[
                    len("text:"):].strip()
                continue

        if message_id is not None:
            observations.append(
                ObservedTaskData(
                    message_id=message_id,
                    writer=writer,
                    text=text,
                )
            )

        return tuple(observations)

    @staticmethod
    def _is_actionable(text: str) -> bool:
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

        return normalized.startswith(action_prefixes)

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


@dataclass(frozen=True, slots=True)
class ObservedTaskData:
    """Internal representation of a parsed Technocore message."""

    message_id: int
    writer: str
    text: str