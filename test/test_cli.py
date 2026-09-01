from __future__ import annotations

from src.agent import cli


def test_cli_requires_task_when_not_autonomous(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["flop-agent"],
    )

    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError(
            "expected argparse failure"
        )


def test_cli_accepts_direct_task(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "flop-agent",
            "Calculate",
            "12",
            "*",
            "8",
        ],
    )

    exit_code = cli.main()

    assert exit_code == 0


def test_cli_autonomous_mode(
    monkeypatch,
) -> None:
    class FakeRunner:
        def close(self):
            pass

        def run_once(self):
            from src.agent.autonomous import AutonomousRun

            return AutonomousRun(
                discovered=(),
                results=(),
            )

    monkeypatch.setattr(
        cli.AutonomousRunner,
        "create",
        lambda **kwargs: FakeRunner(),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "flop-agent",
            "--autonomous",
        ],
    )

    exit_code = cli.main()

    assert exit_code == 0


def test_cli_autonomous_mode_passes_since(
    monkeypatch,
) -> None:
    captured = {}

    class FakeRunner:
        def close(self):
            pass

        def run_once(self):
            from src.agent.autonomous import AutonomousRun

            return AutonomousRun(
                discovered=(),
                results=(),
            )

    def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeRunner()

    monkeypatch.setattr(
        cli.AutonomousRunner,
        "create",
        fake_create,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "flop-agent",
            "--autonomous",
            "--since",
            "12345",
        ],
    )

    exit_code = cli.main()

    assert exit_code == 0
    assert captured["since"] == 12345


def test_cli_autonomous_mode_prints_results_only_for_executed_tasks(
    monkeypatch,
    capsys,
) -> None:
    from src.agent.autonomous import AutonomousRun
    from src.agent.decision import AutonomyAction
    from src.agent.qualification import (
        QualificationCapability,
        QualificationDecision,
        QualificationResult,
    )
    from src.agent.result import ExecutionResult
    from src.agent.task import Task, TaskStatus
    from src.agent.task_source import ObservedTask

    accepted_task = Task(
        description="Calculate 10 + 5",
    )

    rejected_task = Task(
        description="Delete everything",
    )

    ignored_task = Task(
        description="Research autonomous agents",
    )

    discovered = (
        ObservedTask(
            task=accepted_task,
            message_id=109,
            writer="agent-a",
            text=accepted_task.description,
        ),
        ObservedTask(
            task=rejected_task,
            message_id=110,
            writer="agent-b",
            text=rejected_task.description,
        ),
        ObservedTask(
            task=ignored_task,
            message_id=111,
            writer="agent-c",
            text=ignored_task.description,
        ),
    )

    from src.agent.loop import AgentLoopResult

    result = AgentLoopResult(
        task_id=accepted_task.id,
        result=ExecutionResult(
            task_id=accepted_task.id,
            status=TaskStatus.COMPLETED,
            executed_steps=1,
            output="15",
        ),
        iterations=1,
        action=AutonomyAction.COMPLETE,
    )

    class FakeRunner:
        def run_once(self):
            return AutonomousRun(
                discovered=discovered,
                results=(result,),
                qualifications=(
                    QualificationResult(
                        decision=QualificationDecision.ACCEPT,
                        reason="supported",
                        task=accepted_task,
                        capability=QualificationCapability.CALCULATOR,
                    ),
                    QualificationResult(
                        decision=QualificationDecision.REJECT,
                        reason="restricted capability",
                    ),
                    QualificationResult(
                        decision=QualificationDecision.IGNORE,
                        reason="unsupported capability",
                    ),
                ),
            )

        def close(self):
            pass

    monkeypatch.setattr(
        cli.AutonomousRunner,
        "create",
        lambda **kwargs: FakeRunner(),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "flop-agent",
            "--autonomous",
        ],
    )

    exit_code = cli.main()

    assert exit_code == 0

    output = capsys.readouterr().out

    assert "Message: 109" in output
    assert "Task:    Calculate 10 + 5" in output
    assert "Output:  15" in output

    assert "Message: 110" not in output
    assert "Message: 111" not in output
    assert "Delete everything" not in output
    assert "Research autonomous agents" not in output