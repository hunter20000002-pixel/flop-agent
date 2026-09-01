from __future__ import annotations

import argparse

from src.agent.daemon import AutonomousDaemon
from src.agent.loop import AgentLoop
from src.agent.runner import AutonomousRunner
from src.agent.task import Task


def _build_parser() -> argparse.ArgumentParser:
    """Build the FLOP Agent command-line parser."""

    parser = argparse.ArgumentParser(
        prog="flop-agent",
        description="Run the FLOP autonomous agent.",
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--autonomous",
        action="store_true",
        help="Poll Technocore and execute one discovered-task cycle.",
    )

    mode.add_argument(
        "--daemon",
        action="store_true",
        help="Run the autonomous Technocore worker continuously.",
    )

    parser.add_argument(
        "task",
        nargs="*",
        help="Task for the agent to execute.",
    )

    parser.add_argument(
        "--since",
        type=int,
        default=0,
        help=(
            "Only process Technocore messages after "
            "this sequence number."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "SQLite checkpoint database for persistent "
            "Technocore progress."
        ),
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Seconds to wait between daemon cycles.",
    )

    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Stop the daemon after this many cycles.",
    )

    return parser


def _print_autonomous_run(run) -> None:
    """Print the results of one autonomous execution cycle."""

    print()
    print("=== FLOP AUTONOMOUS AGENT ===")
    print(f"Discovered: {len(run.discovered)}")
    print(f"Executed:   {len(run.results)}")

    if not run.results:
        return

    if not run.qualifications:
        return

    accepted = tuple(
        (
            observed,
            qualification,
        )
        for observed, qualification in zip(
            run.discovered,
            run.qualifications,
        )
        if qualification.accepted
    )

    if len(accepted) != len(run.results):
        raise RuntimeError(
            "autonomous run contains a result/qualification mismatch"
        )

    for (observed, _qualification), result in zip(
        accepted,
        run.results,
    ):
        execution = result.result

        print()
        print(f"Message: {observed.message_id}")
        print(f"Writer:  {observed.writer}")
        print(f"Task:    {observed.task.description}")
        print(f"Status:  {execution.status.value}")

        if execution.output is not None:
            print(f"Output:  {execution.output}")

        if execution.error is not None:
            print(f"Error:   {execution.error}")


def _run_single_autonomous_cycle(
    args: argparse.Namespace,
) -> int:
    """Run exactly one autonomous Technocore cycle."""

    runner = AutonomousRunner.create(
        since=args.since,
        checkpoint_path=args.checkpoint,
    )

    try:
        run = runner.run_once()

        _print_autonomous_run(run)

        return (
            0
            if all(
                result.result.succeeded
                for result in run.results
            )
            else 1
        )

    finally:
        runner.close()


def _run_daemon(
    args: argparse.Namespace,
) -> int:
    """Run the persistent autonomous Technocore daemon."""

    runner = AutonomousRunner.create(
        since=args.since,
        checkpoint_path=args.checkpoint,
    )

    daemon = AutonomousDaemon(
        runner,
        interval=args.interval,
    )

    def on_cycle(cycle) -> None:
        print()
        print("=== FLOP AUTONOMOUS CYCLE ===")
        print(f"Discovered: {cycle.discovered}")
        print(f"Completed:  {cycle.completed}")
        print(f"Failed:     {cycle.failed}")

        if cycle.error is not None:
            print(f"Error:      {cycle.error}")

    try:
        daemon.run_forever(
            max_cycles=args.max_cycles,
            on_cycle=on_cycle,
        )

        return 0

    except KeyboardInterrupt:
        print()
        print("FLOP Agent daemon stopped.")

        return 0

    finally:
        daemon.close()


def _run_direct_task(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Execute one directly supplied task."""

    if not args.task:
        parser.error(
            "provide a task or use --autonomous or --daemon"
        )

    description = " ".join(args.task)
    task = Task(description=description)

    loop = AgentLoop()
    result = loop.run(task)

    print()
    print("=== FLOP AGENT ===")
    print(f"Task:       {description}")
    print(f"Task ID:    {result.task_id}")
    print(f"Iterations: {result.iterations}")
    print(f"Action:     {result.action.value}")
    print(f"Status:     {result.result.status.value}")

    if result.result.output is not None:
        print()
        print("Output:")
        print(result.result.output)

    if result.result.error is not None:
        print()
        print("Error:")
        print(result.result.error)

    return (
        0
        if result.result.succeeded
        else 1
    )


def main() -> int:
    """Run the FLOP Agent command-line interface."""

    parser = _build_parser()
    args = parser.parse_args()

    if args.interval < 0:
        parser.error(
            "--interval cannot be negative"
        )

    if args.max_cycles is not None and args.max_cycles <= 0:
        parser.error(
            "--max-cycles must be greater than zero"
        )

    if args.since < 0:
        parser.error(
            "--since cannot be negative"
        )

    if args.daemon:
        if args.task:
            parser.error(
                "tasks cannot be supplied with --daemon"
            )

        return _run_daemon(args)

    if args.autonomous:
        if args.task:
            parser.error(
                "tasks cannot be supplied with --autonomous"
            )

        return _run_single_autonomous_cycle(args)

    if args.checkpoint is not None:
        parser.error(
            "--checkpoint requires --autonomous or --daemon"
        )

    if args.max_cycles is not None:
        parser.error(
            "--max-cycles requires --daemon"
        )

    if args.interval != 30.0:
        parser.error(
            "--interval requires --daemon"
        )

    return _run_direct_task(
        args,
        parser,
    )


if __name__ == "__main__":
    raise SystemExit(main())