from types import SimpleNamespace

from open_er.server.rubrics import grade_episode
from open_er.server.tasks import get_task_config


def test_grade_episode_is_normalized():
    task = get_task_config("easy_single_critical")
    patient = SimpleNamespace(
        true_esi=1,
        terminal_outcome="admitted",
        unsafe_disposition=False,
        urgent_wait_breached=False,
        required_tests={"ecg", "troponin"},
        ordered_tests={"ecg", "troponin"},
        unnecessary_tests=set(),
        correct_disposition_hit=True,
    )
    result = grade_episode(task, [patient])
    assert 0.0 <= result.score <= 1.0
    assert all(0.0 <= value <= 1.0 for value in result.breakdown.values())
