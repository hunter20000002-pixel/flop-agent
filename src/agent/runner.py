from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agent.autonomous import AutonomousAgent, AutonomousRun
from src.agent.checkpoint_store import SQLiteTaskCheckpointStore
from src.agent.loop import AgentLoop
from src.agent.publisher import TechnocoreResultPublisher
from src.agent.task_source import TechnocoreTaskSource
from src.config import Config, DEFAULT_CONFIG


@dataclass(frozen=True, slots=True)
class AutonomousRunner:
    """Production wiring for the FLOP autonomous agent."""

    agent: AutonomousAgent
    task_source: TechnocoreTaskSource
    loop: AgentLoop
    checkpoint_store: SQLiteTaskCheckpointStore | None = None
    publisher: TechnocoreResultPublisher | None = None

    @classmethod
    def create(
        cls,
        *,
        config: Config = DEFAULT_CONFIG,
        since: int = 0,
        max_iterations: int = 10,
        max_retries: int = 3,
        task_source: TechnocoreTaskSource | None = None,
        checkpoint_path: str | Path | None = None,
        publisher: TechnocoreResultPublisher | None = None,
        enable_publishing: bool = True,
    ) -> AutonomousRunner:
        """Create a fully configured autonomous FLOP runner."""

        checkpoint_store: SQLiteTaskCheckpointStore | None = None

        if checkpoint_path is not None:
            checkpoint_store = SQLiteTaskCheckpointStore(
                checkpoint_path,
            )

        if task_source is None:
            task_source = TechnocoreTaskSource(
                room=config.room,
                since=since,
                checkpoint_store=checkpoint_store,
            )

        loop = AgentLoop(
            max_iterations=max_iterations,
            max_retries=max_retries,
        )

        if (
            publisher is None
            and enable_publishing
        ):
            publisher = TechnocoreResultPublisher(
                config=config,
            )

        agent = AutonomousAgent(
            task_source=task_source,
            runtime=loop,
            publisher=publisher,
        )

        return cls(
            agent=agent,
            task_source=task_source,
            loop=loop,
            checkpoint_store=checkpoint_store,
            publisher=publisher,
        )

    def run_once(self) -> AutonomousRun:
        """
        Poll Technocore once and execute all discovered tasks.

        The runner preserves the AutonomousAgent result contract
        without transforming or unwrapping execution results.
        """

        return self.agent.run_once()

    def close(self) -> None:
        """Close persistent resources owned by the runner."""

        if self.checkpoint_store is not None:
            self.checkpoint_store.close()

    def __enter__(
        self,
    ) -> AutonomousRunner:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
