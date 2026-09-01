from __future__ import annotations

from src.tools.calculator import CalculatorTool
from src.tools.filesystem import FilesystemTool
from src.tools.registry import ToolRegistry
from src.tools.technocore import TechnocoreObserverTool


def create_builtin_registry() -> ToolRegistry:
    """Create a registry containing FLOP Agent's built-in tools."""

    registry = ToolRegistry()

    registry.register(CalculatorTool())
    registry.register(FilesystemTool())
    registry.register(TechnocoreObserverTool())

    return registry