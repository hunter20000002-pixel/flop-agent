import pytest

from src.tools.base import Tool, ToolResult
from src.tools.registry import ToolRegistry


class ExampleTool(Tool):
    @property
    def name(self) -> str:
        return "example"

    @property
    def description(self) -> str:
        return "An example tool."

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            output=kwargs.get("value"),
        )


class AnotherTool(Tool):
    @property
    def name(self) -> str:
        return "another"

    @property
    def description(self) -> str:
        return "Another example tool."

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            output=kwargs,
        )


def test_registry_starts_empty():
    registry = ToolRegistry()

    assert len(registry) == 0
    assert registry.list_tools() == ()


def test_registry_registers_tool():
    registry = ToolRegistry()
    tool = ExampleTool()

    registry.register(tool)

    assert len(registry) == 1
    assert registry.has("example")
    assert registry.get("example") is tool


def test_registry_can_register_multiple_tools():
    registry = ToolRegistry()

    first = ExampleTool()
    second = AnotherTool()

    registry.register(first)
    registry.register(second)

    assert len(registry) == 2
    assert registry.list_tools() == (first, second)


def test_registry_rejects_duplicate_tool_names():
    registry = ToolRegistry()

    registry.register(ExampleTool())

    with pytest.raises(ValueError, match="tool already registered"):
        registry.register(ExampleTool())


def test_registry_rejects_invalid_tool():
    registry = ToolRegistry()

    with pytest.raises(TypeError, match="tool must be a Tool"):
        registry.register("not a tool")


def test_registry_rejects_empty_tool_name_lookup():
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="tool name cannot be empty"):
        registry.get("   ")


def test_registry_raises_for_missing_tool():
    registry = ToolRegistry()

    with pytest.raises(KeyError, match="tool not found"):
        registry.get("missing")


def test_registry_has_returns_false_for_missing_tool():
    registry = ToolRegistry()

    assert not registry.has("missing")