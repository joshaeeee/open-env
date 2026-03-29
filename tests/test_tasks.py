import pytest

from open_er.server.tasks import OFFICIAL_TASKS, get_task_config


def test_official_tasks_present():
    assert set(OFFICIAL_TASKS) == {
        "easy_single_critical",
        "medium_evening_rush",
        "hard_capacity_crunch",
    }


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        get_task_config("unknown")
