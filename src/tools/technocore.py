from __future__ import annotations

from typing import Any

from src.agent.observation import TechnocoreObservation
from src.client import read_since
from src.config import DEFAULT_CONFIG
from src.parser import parse_messages
from src.tools.base import Tool, ToolResult


class TechnocoreObserverTool(Tool):
    """Read and structure new messages from a Technocore room."""

    @property
    def name(self) -> str:
        return "technocore_observer"

    @property
    def description(self) -> str:
        return (
            "Observe new Technocore messages after a sequence number "
            "and return a structured observation."
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """Fetch and parse Technocore messages."""

        try:
            room = kwargs.get("room", DEFAULT_CONFIG.room)
            since = kwargs.get("since", 0)
            base_url = kwargs.get(
                "base_url",
                DEFAULT_CONFIG.base_url,
            )
            user_agent = kwargs.get(
                "user_agent",
                DEFAULT_CONFIG.user_agent,
            )

            if not isinstance(room, str):
                raise TypeError("room must be a string")

            if not room.strip():
                raise ValueError("room must not be empty")

            if not isinstance(since, int):
                raise TypeError("since must be an integer")

            if since < 0:
                raise ValueError(
                    "since must be greater than or equal to zero"
                )

            if not isinstance(base_url, str):
                raise TypeError("base_url must be a string")

            if not isinstance(user_agent, str):
                raise TypeError("user_agent must be a string")

            body = read_since(
                base_url,
                room,
                since,
                user_agent,
            )

            messages = tuple(parse_messages(body))

            observation = TechnocoreObservation.from_messages(
                room=room,
                since=since,
                messages=messages,
            )

            return ToolResult(
                success=True,
                output=observation.to_untrusted_text(),
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )