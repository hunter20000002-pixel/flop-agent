from __future__ import annotations

import argparse

from src.agent.loop import AgentLoop
from src.agent.task import Task


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="flop-agent",
        description="Run the FLOP autonomous agent.",
    )

    parser.add_argument(
        "task",
        nargs="+",
        help="Task for the agent to execute.",
    )

    args = parser.parse_args()

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
