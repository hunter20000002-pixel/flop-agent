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