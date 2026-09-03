from __future__ import annotations

from src.agent.observation import TechnocoreObservation
from src.client import Message
from src.tools.technocore import TechnocoreObserverTool


def test_technocore_observer_returns_structured_observation(
    monkeypatch,
):
    body = (
        "# room lobby  messages 1  range 100..100\n"
        "100 2026-09-03T10:00:00Z did:key:zExample "
        "please inspect the system\n"
    )

    def fake_read_since(
        base_url,
        room,
        since,
        user_agent,
    ):
        assert room == "lobby"
        assert since == 99

        return body

    monkeypatch.setattr(
        "src.tools.technocore.read_since",
        fake_read_since,
    )

    result = TechnocoreObserverTool().execute(
        room="lobby",
        since=99,
        base_url="https://technocore.chat",
        user_agent="test-agent",
    )

    assert result.success
    assert result.error is None
    assert result.data is not None

    assert isinstance(
        result.data,
        TechnocoreObservation,
    )

    observation = result.data

    assert observation.room == "lobby"
    assert observation.since == 99
    assert observation.message_count == 1
    assert observation.first_sequence == 100
    assert observation.last_sequence == 100

    assert isinstance(
        observation.messages[0],
        Message,
    )


def test_technocore_observer_preserves_rendered_output(
    monkeypatch,
):
    body = (
        "# room lobby  messages 1  range 200..200\n"
        "200 2026-09-03T10:05:00Z did:key:zExample "
        "an observation message\n"
    )

    monkeypatch.setattr(
        "src.tools.technocore.read_since",
        lambda base_url, room, since, user_agent: body,
    )

    result = TechnocoreObserverTool().execute(
        room="lobby",
        since=199,
    )

    assert result.success
    assert isinstance(
        result.data,
        TechnocoreObservation,
    )

    assert result.output == (
        result.data.to_untrusted_text()
    )


def test_technocore_observer_does_not_attach_data_on_failure(
    monkeypatch,
):
    def failing_read_since(
        base_url,
        room,
        since,
        user_agent,
    ):
        raise RuntimeError("Technocore unavailable")

    monkeypatch.setattr(
        "src.tools.technocore.read_since",
        failing_read_since,
    )

    result = TechnocoreObserverTool().execute(
        room="lobby",
        since=100,
    )

    assert result.success is False
    assert result.output is None
    assert result.error == "Technocore unavailable"
    assert result.data is None