from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tools.base import Tool, ToolResult


class FilesystemTool(Tool):
    """Read basic filesystem information for the agent."""

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return (
            "Inspect files and directories. "
            "Supports listing a directory and reading a text file."
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs.get("operation")
        path_value = kwargs.get("path")

        if not isinstance(operation, str):
            return ToolResult(
                success=False,
                error="operation is required",
            )

        if not isinstance(path_value, str):
            return ToolResult(
                success=False,
                error="path is required",
            )

        path = Path(path_value)

        try:
            if operation == "list":
                if not path.exists():
                    return ToolResult(
                        success=False,
                        error=f"path does not exist: {path}",
                    )

                if not path.is_dir():
                    return ToolResult(
                        success=False,
                        error=f"path is not a directory: {path}",
                    )

                entries = sorted(
                    str(entry)
                    for entry in path.iterdir()
                )

                return ToolResult(
                    success=True,
                    output=entries,
                )

            if operation == "read":
                if not path.exists():
                    return ToolResult(
                        success=False,
                        error=f"path does not exist: {path}",
                    )

                if not path.is_file():
                    return ToolResult(
                        success=False,
                        error=f"path is not a file: {path}",
                    )

                return ToolResult(
                    success=True,
                    output=path.read_text(
                        encoding="utf-8"
                    ),
                )

            return ToolResult(
                success=False,
                error=f"unsupported filesystem operation: {operation}",
            )

        except OSError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )
