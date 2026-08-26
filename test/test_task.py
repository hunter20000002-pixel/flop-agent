from datetime import datetime, timezone
from uuid import UUID

import pytest

from src.agent.task import Task, TaskStatus


def test_task_has_valid_defaults():
    task = Task(description="Test autonomous task")

    assert isinstance(task.id, UUID)
    assert task.description == "Test autonomous task"
    assert task.status == TaskStatus.PENDING
    assert task.created_at.tzinfo == timezone.utc
    assert task.updated_at.tzinfo == timezone.utc


def test_task_lifecycle():
    task = Task(description="Run a test task")

    task.mark_planning()
    assert task.status == TaskStatus.PLANNING

    task.mark_ready()
    assert task.status == TaskStatus.READY

    task.mark_running()
    assert task.status == TaskStatus.RUNNING

    task.mark_completed()
    assert task.status == TaskStatus.COMPLETED


def test_task_can_fail():
    task = Task(description="Failing task")

    task.mark_running()
    task.mark_failed()

    assert task.status == TaskStatus.FAILED


def test_task_can_be_cancelled():
    task = Task(description="Cancelled task")

    task.mark_cancelled()

    assert task.status == TaskStatus.CANCELLED


def test_updated_at_changes_when_status_changes():
    task = Task(description="Timestamp test")
    original_updated_at = task.updated_at

    task.mark_running()

    assert task.updated_at >= original_updated_at


def test_invalid_status_raises_type_error():
    task = Task(description="Invalid status")

    with pytest.raises(TypeError):
        task.set_status("running")