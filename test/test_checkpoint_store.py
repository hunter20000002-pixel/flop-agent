
from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.checkpoint_store import SQLiteTaskCheckpointStore


def test_checkpoint_store_returns_default_since(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        assert store.get_since(
            "technocore:lobby",
        ) == 0


def test_checkpoint_store_persists_since(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        store.set_since(
            "technocore:lobby",
            500,
        )

        assert store.get_since(
            "technocore:lobby",
        ) == 500

    with SQLiteTaskCheckpointStore(database) as store:
        assert store.get_since(
            "technocore:lobby",
        ) == 500


def test_checkpoint_store_updates_since(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        store.set_since(
            "technocore:lobby",
            500,
        )

        store.set_since(
            "technocore:lobby",
            600,
        )

        assert store.get_since(
            "technocore:lobby",
        ) == 600


def test_checkpoint_store_tracks_multiple_sources(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        store.set_since(
            "technocore:lobby",
            500,
        )

        store.set_since(
            "technocore:test",
            900,
        )

        assert store.get_since(
            "technocore:lobby",
        ) == 500

        assert store.get_since(
            "technocore:test",
        ) == 900


def test_checkpoint_store_marks_new_message_processed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        assert store.mark_processed(
            "technocore:lobby",
            500,
        )

        assert store.is_processed(
            "technocore:lobby",
            500,
        )


def test_checkpoint_store_rejects_duplicate_message(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        assert store.mark_processed(
            "technocore:lobby",
            500,
        )

        assert not store.mark_processed(
            "technocore:lobby",
            500,
        )


def test_checkpoint_store_separates_sources(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        assert store.mark_processed(
            "technocore:lobby",
            500,
        )

        assert not store.mark_processed(
            "technocore:lobby",
            500,
        )

        assert store.mark_processed(
            "technocore:test",
            500,
        )


def test_checkpoint_store_rejects_invalid_source(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        with pytest.raises(
            TypeError,
            match="source must be a string",
        ):
            store.get_since(123)  # type: ignore[arg-type]

        with pytest.raises(
            ValueError,
            match="source cannot be empty",
        ):
            store.get_since("")


def test_checkpoint_store_rejects_invalid_since(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        with pytest.raises(
            TypeError,
            match="since must be an integer",
        ):
            store.set_since(
                "technocore:lobby",
                "500",  # type: ignore[arg-type]
            )

        with pytest.raises(
            ValueError,
            match="since cannot be negative",
        ):
            store.set_since(
                "technocore:lobby",
                -1,
            )


def test_checkpoint_store_rejects_invalid_message_id(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.db"

    with SQLiteTaskCheckpointStore(database) as store:
        with pytest.raises(
            TypeError,
            match="message_id must be an integer",
        ):
            store.mark_processed(
                "technocore:lobby",
                "500",  # type: ignore[arg-type]
            )

        with pytest.raises(
            ValueError,
            match="message_id cannot be negative",
        ):
            store.mark_processed(
                "technocore:lobby",
                -1,
            )
