from __future__ import annotations

from src.tools.base import Tool


class ToolRegistry:
    """Registry of tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by its unique name."""

        if not isinstance(tool, Tool):
            raise TypeError("tool must be a Tool")

        if not tool.name.strip():
            raise ValueError("tool name cannot be empty")

        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return a registered tool by name."""

        if not name.strip():
            raise ValueError("tool name cannot be empty")

        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"tool not found: {name}") from None

    def has(self, name: str) -> bool:
        """Return True when a tool is registered."""

        return name in self._tools

    def list_tools(self) -> tuple[Tool, ...]:
        """Return all registered tools."""

        return tuple(self._tools.values())

    def __len__(self) -> int:
        """Return the number of registered tools."""

        return len(self._tools)