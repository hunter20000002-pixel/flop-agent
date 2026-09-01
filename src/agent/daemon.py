from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Protocol

from src.agent.autonomous import AutonomousRun


SleepFunction = Callable[[float], None]
CycleCallback = Callable[["DaemonCycle"], None]


class Runner(Protocol):
    """Interface required by the autonomous daemon."""

    def run_once(self) -> AutonomousRun:
        """Execute one autonomous observation/execution cycle."""
        ...

    def close(self) -> None:
        """Release runner resources."""
        ...


@dataclass(frozen=True, slots=True)
class DaemonCycle:
    """Result of one autonomous daemon cycle."""

    discovered: int
    completed: int
    failed: int
    error: str | None = None


class AutonomousDaemon:
    """
    Persistent worker for the FLOP autonomous agent.

    The daemon repeatedly:

        poll Technocore
        -> execute discovered tasks
        -> publish results
        -> acknowledge successful tasks
        -> sleep
        -> repeat

    The daemon owns the long-running lifecycle while the runner
    remains responsible for autonomous execution.
    """

    def __init__(
        self,
        runner: Runner,
        *,
        interval: float = 30.0,
        sleep: SleepFunction = time.sleep,
    ) -> None:
        if not hasattr(runner, "run_once"):
            raise TypeError(
                "runner must provide a run_once() method"
            )

        if not hasattr(runner, "close"):
            raise TypeError(
                "runner must provide a close() method"
            )

        if not isinstance(interval, (int, float)):
            raise TypeError(
                "interval must be a number"
            )

        if interval < 0:
            raise ValueError(
                "interval cannot be negative"
            )

        if not callable(sleep):
            raise TypeError(
                "sleep must be callable"
            )

        self.runner = runner
        self.interval = float(interval)
        self.sleep = sleep
        self._running = False

    @property
    def running(self) -> bool:
        """Return True while the daemon is running."""
        return self._running

    def stop(self) -> None:
        """Request graceful shutdown."""
        self._running = False

    def run_cycle(self) -> DaemonCycle:
        """
        Execute exactly one autonomous cycle.

        Runner failures are captured and returned as cycle errors so
        the daemon can continue operating on the next cycle.
        """

        try:
            autonomous_run = self.runner.run_once()

        except Exception as exc:
            return DaemonCycle(
                discovered=0,
                completed=0,
                failed=0,
                error=str(exc),
            )

        completed = 0
        failed = 0

        for loop_result in autonomous_run.results:
            if loop_result.result.succeeded:
                completed += 1
            else:
                failed += 1

        return DaemonCycle(
            discovered=len(
                autonomous_run.discovered
            ),
            completed=completed,
            failed=failed,
        )

    def run_forever(
        self,
        *,
        max_cycles: int | None = None,
        on_cycle: CycleCallback | None = None,
    ) -> None:
        """
        Run the autonomous worker continuously.

        max_cycles is primarily useful for tests and controlled
        development runs. When None, the daemon continues until
        stop() is called or the process is terminated.
        """

        if max_cycles is not None:
            if not isinstance(max_cycles, int):
                raise TypeError(
                    "max_cycles must be an integer or None"
                )

            if max_cycles <= 0:
                raise ValueError(
                    "max_cycles must be greater than zero"
                )

        if on_cycle is not None and not callable(on_cycle):
            raise TypeError(
                "on_cycle must be callable or None"
            )

        self._running = True
        cycles = 0

        try:
            while self._running:
                cycle = self.run_cycle()
                cycles += 1

                if on_cycle is not None:
                    on_cycle(cycle)

                if (
                    max_cycles is not None
                    and cycles >= max_cycles
                ):
                    break

                if not self._running:
                    break

                self.sleep(self.interval)

        finally:
            self._running = False

    def close(self) -> None:
        """Stop the daemon and close the underlying runner."""
        self.stop()
        self.runner.close()

    def __enter__(
        self,
    ) -> AutonomousDaemon:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()