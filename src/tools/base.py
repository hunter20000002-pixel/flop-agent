from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured result returned by a tool execution."""

    success: bool
    output: Any = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        """Return True when the tool execution failed."""

        return not self.success


class Tool(ABC):
    """Base interface for tools available to the agent."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique tool name."""

        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of the tool."""

        raise NotImplementedError

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the supplied arguments."""

        raise NotImplementedError