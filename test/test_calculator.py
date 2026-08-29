import pytest

from src.tools.calculator import CalculatorTool


def test_calculator_returns_result():
    result = CalculatorTool().execute(
        expression="2 + 2",
    )

    assert result.success
    assert result.output == 4


def test_calculator_handles_complex_expression():
    result = CalculatorTool().execute(
        expression="10 * 5 + 2",
    )

    assert result.success
    assert result.output == 52


def test_calculator_handles_division():
    result = CalculatorTool().execute(
        expression="10 / 4",
    )

    assert result.success
    assert result.output == 2.5


def test_calculator_rejects_empty_expression():
    result = CalculatorTool().execute(
        expression="   ",
    )

    assert not result.success


def test_calculator_rejects_non_string_expression():
    result = CalculatorTool().execute(
        expression=42,
    )

    assert not result.success


def test_calculator_does_not_execute_arbitrary_code():
    result = CalculatorTool().execute(
        expression="__import__('os').system('echo hacked')",
    )

    assert not result.success