from __future__ import annotations

import ast
import operator
from typing import Any

from src.tools.base import Tool, ToolResult


class CalculatorTool(Tool):
    """Safely evaluate basic mathematical expressions."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate a basic mathematical expression."

    def execute(self, **kwargs: Any) -> ToolResult:
        expression = kwargs.get("expression")

        if not isinstance(expression, str):
            return ToolResult(
                success=False,
                error="expression must be a string",
            )

        expression = expression.strip()

        if not expression:
            return ToolResult(
                success=False,
                error="expression cannot be empty",
            )

        try:
            tree = ast.parse(
                expression,
                mode="eval",
            )

            result = self._evaluate(tree.body)

            return ToolResult(
                success=True,
                output=result,
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

    def _evaluate(self, node: ast.AST) -> int | float:
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv,
        }

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("unsupported constant")

        if isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand)

            if isinstance(node.op, ast.USub):
                return -value

            if isinstance(node.op, ast.UAdd):
                return +value

            raise ValueError("unsupported unary operator")

        if isinstance(node, ast.BinOp):
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            operation = operators.get(type(node.op))

            if operation is None:
                raise ValueError("unsupported operator")

            return operation(left, right)

        raise ValueError("unsupported expression")