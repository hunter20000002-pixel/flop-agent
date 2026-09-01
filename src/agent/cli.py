from __future__ import annotations

import argparse

from src.agent.loop import AgentLoop
from src.agent.runner import AutonomousRunner
from src.agent.task import Task


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="flop-agent",
        description="Run the FLOP autonomous agent.",
    )

    parser.add_argument(
        "task",
        nargs="*",
        help="Task for the agent to execute.",
    )

    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Poll Technocore and execute discovered tasks.",
    )

    parser.add_argument(
        "--since",
        type=int,
        default=0,
        help="Only process Technocore messages after this sequence number.",
    )

    args = parser.parse_args()

    if args.autonomous:
        runner = AutonomousRunner.create(
            since=args.since,
        )

        run = runner.run_once()

        print()
        print("=== FLOP AUTONOMOUS AGENT ===")
        print(f"Discovered: {len(run.discovered)}")
        print(f"Executed:   {len(run.results)}")

        for observed, result in zip(
            run.discovered,
            run.results,
        ):
            print()
            print(f"Message: {observed.message_id}")
            print(f"Writer:  {observed.writer}")
            print(f"Task:    {observed.task.description}")
            print(f"Status:  {result.status.value}")

            if result.output is not None:
                print(f"Output:  {result.output}")

            if result.error is not None:
                print(f"Error:   {result.error}")

        return 0 if all(
            result.succeeded
            for result in run.results
        ) else 1

    if not args.task:
        parser.error(
            "provide a task or use --autonomous"
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

    return 0 if result.result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())