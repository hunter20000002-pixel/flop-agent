from pathlib import Path

from src.tools.base import ToolResult
from src.tools.filesystem import FilesystemTool


def test_filesystem_tool_exposes_metadata():
    tool = FilesystemTool()

    assert tool.name == "filesystem"
    assert "files" in tool.description.lower()


def test_filesystem_tool_lists_directory(tmp_path: Path):
    (tmp_path / "alpha.txt").write_text(
        "alpha",
        encoding="utf-8",
    )
    (tmp_path / "beta.txt").write_text(
        "beta",
        encoding="utf-8",
    )

    result = FilesystemTool().execute(
        operation="list",
        path=str(tmp_path),
    )

    assert isinstance(result, ToolResult)
    assert result.success
    assert result.output == [
        str(tmp_path / "alpha.txt"),
        str(tmp_path / "beta.txt"),
    ]


def test_filesystem_tool_reads_file(tmp_path: Path):
    file_path = tmp_path / "example.txt"

    file_path.write_text(
        "hello from FLOP",
        encoding="utf-8",
    )

    result = FilesystemTool().execute(
        operation="read",
        path=str(file_path),
    )

    assert result.success
    assert result.output == "hello from FLOP"


def test_filesystem_tool_rejects_missing_path(tmp_path: Path):
    result = FilesystemTool().execute(
        operation="read",
        path=str(tmp_path / "missing.txt"),
    )

    assert not result.success
    assert "does not exist" in result.error


def test_filesystem_tool_rejects_invalid_operation(tmp_path: Path):
    result = FilesystemTool().execute(
        operation="delete",
        path=str(tmp_path),
    )

    assert not result.success
    assert "unsupported" in result.error
