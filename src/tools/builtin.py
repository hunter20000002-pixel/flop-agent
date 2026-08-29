from __future__ import annotations

from src.tools.calculator import CalculatorTool
from src.tools.registry import ToolRegistry


def create_builtin_registry() -> ToolRegistry:
    """Create a registry containing FLOP's built-in tools."""

    registry = ToolRegistry()

    registry.register(CalculatorTool())

    return registry