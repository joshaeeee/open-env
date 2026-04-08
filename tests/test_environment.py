from open_er.models import ERAction
from open_er.server.environment import ERTriageEnvironment


def test_reset_is_reproducible_for_same_seed():
    env_a = ERTriageEnvironment()
    env_b = ERTriageEnvironment()
    obs_a = env_a.reset(task_id="easy_single_critical", seed=11)
    obs_b = env_b.reset(task_id="easy_single_critical", seed=11)
    assert [patient.patient_id for patient in obs_a.patients] == [patient.patient_id for patient in obs_b.patients]
    assert [patient.chief_complaint for patient in obs_a.patients] == [patient.chief_complaint for patient in obs_b.patients]


def test_invalid_patient_command_does_not_crash():
    env = ERTriageEnvironment()
    env.reset(task_id="easy_single_critical", seed=11)
    obs = env.step(ERAction())
    assert obs.done is False
    obs = env.step(ERAction(commands=[]))
    assert obs.done is False


def test_terminal_score_exists_at_horizon():
    env = ERTriageEnvironment()
    env.reset(task_id="easy_single_critical", seed=11)
    result = None
    for _ in range(env.state.step_count, 96):
        result = env.step(ERAction(commands=[]))
    assert result is not None
    assert result.done is True
    assert 0.0 <= float(env.state.benchmark_score or 0.0) <= 1.0


def test_state_does_not_leak_hidden_diagnosis():
    env = ERTriageEnvironment()
    env.reset(task_id="easy_single_critical", seed=11)
    payload = env.state.model_dump()
    serialized = str(payload)
    assert "diagnosis" not in serialized
    assert "required_tests" not in serialized
