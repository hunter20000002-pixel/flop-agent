from __future__ import annotations

from dataclasses import dataclass

from src.agent.autonomous import AutonomousAgent, AutonomousRun
from src.agent.loop import AgentLoop
from src.agent.result import ExecutionResult
from src.agent.task_source import TechnocoreTaskSource
from src.config import Config, DEFAULT_CONFIG


@dataclass(frozen=True, slots=True)
class AutonomousRunner:
    """Production wiring for the FLOP autonomous agent."""

    agent: AutonomousAgent
    task_source: TechnocoreTaskSource
    loop: AgentLoop

    @classmethod
    def create(
        cls,
        *,
        config: Config = DEFAULT_CONFIG,
        since: int = 0,
        max_iterations: int = 10,
        max_retries: int = 3,
        task_source: TechnocoreTaskSource | None = None,
    ) -> AutonomousRunner:
        """Create a fully configured autonomous FLOP runner."""

        if task_source is None:
            task_source = TechnocoreTaskSource(
                room=config.room,
                since=since,
            )

        loop = AgentLoop(
            max_iterations=max_iterations,
            max_retries=max_retries,
        )

        agent = AutonomousAgent(
            task_source=task_source,
            runtime=loop,
        )

        return cls(
            agent=agent,
            task_source=task_source,
            loop=loop,
        )

    def run_once(self) -> AutonomousRun:
        """Poll Technocore once and execute all discovered tasks."""

        autonomous_run = self.agent.run_once()

        results = tuple(
            result.result
            for result in autonomous_run.results
        )

        return AutonomousRun(
            discovered=autonomous_run.discovered,
            results=results,
        )