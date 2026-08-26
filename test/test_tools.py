import pytest

from src.tools.base import Tool, ToolResult


class ExampleTool(Tool):
    @property
    def name(self) -> str:
        return "example"

    @property
    def description(self) -> str:
        return "An example tool for testing."

    def execute(self, **kwargs):
        return ToolResult(
            success=True,
            output=kwargs.get("value"),
        )


class FailingTool(Tool):
    @property
    def name(self) -> str:
        return "failing"

    @property
    def description(self) -> str:
        return "A tool that returns a failure."

    def execute(self, **kwargs):
        return ToolResult(
            success=False,
            error="Tool execution failed.",
        )


def test_tool_exposes_name_and_description():
    tool = ExampleTool()

    assert tool.name == "example"
    assert tool.description == "An example tool for testing."


def test_tool_returns_structured_result():
    tool = ExampleTool()

    result = tool.execute(value="hello")

    assert isinstance(result, ToolResult)
    assert result.success
    assert not result.failed
    assert result.output == "hello"
    assert result.error is None


def test_failed_tool_result():
    result = FailingTool().execute()

    assert isinstance(result, ToolResult)
    assert not result.success
    assert result.failed
    assert result.output is None
    assert result.error == "Tool execution failed."


def test_tool_requires_implementation():
    with pytest.raises(TypeError):
        Tool()